"""Offline test (docs/PRIVACY.md): the full app works with no network.

Disconnect from the network first (turn off Wi-Fi / unplug Ethernet), then run:

    python -m scripts.offline_check

It refuses to run while the internet is reachable. Docker and Ollama must be
running. It uploads a test PDF, asks questions (answer, follow-up, cache, an
out-of-scope question), and deletes the PDF, all in a temporary schema. While it
runs it records every TCP connection owned by the app and Ollama processes.

    python -m scripts.offline_check --with-network

runs the same flow while online and reports any connection to a non-local
address, as evidence that nothing leaves the machine even when it could.

Results are printed and saved to storage/privacy_checks/.
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.check_utils import ROOT, Report, isolated_app, make_pdf

WATCHED_PROCESSES = ("python", "ollama", "llama-server")
# Ollama's system-tray program checks ollama.com for updates. It never receives
# documents or questions (those go to the model server, ollama.exe, on 127.0.0.1),
# so its connections are reported separately instead of failing the check.
TRAY_APP = "ollama app.exe"
LOCAL_PREFIXES = ("127.", "[::1]", "::1", "0.0.0.0", "[::]", "*")


def internet_reachable() -> list[str]:
    """Ways the internet is still reachable (empty list = offline)."""
    reachable = []
    try:
        socket.getaddrinfo("github.com", 443)
        reachable.append("DNS lookup of github.com succeeded")
    except OSError:
        pass
    for address in ("1.1.1.1", "8.8.8.8"):
        try:
            with socket.create_connection((address, 443), timeout=3):
                reachable.append(f"TCP connection to {address}:443 succeeded")
        except OSError:
            pass
    return reachable


class ConnectionWatcher(threading.Thread):
    """Samples `netstat -ano` and keeps connections owned by watched processes."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.remote: set[str] = set()
        self.samples = 0

    @staticmethod
    def watched_pids() -> dict[str, str]:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        pids = {}
        for line in out.splitlines():
            parts = [part.strip('"') for part in line.split('","')]
            if len(parts) >= 2 and parts[0].lower().startswith(WATCHED_PROCESSES):
                pids[parts[1]] = parts[0]
        return pids

    def run(self) -> None:
        while not self.stop_event.is_set():
            pids = self.watched_pids()
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] == "TCP" and parts[4] in pids:
                    remote = parts[2]
                    if not remote.startswith(LOCAL_PREFIXES):
                        self.remote.add(f"{pids[parts[4]]} (pid {parts[4]}) -> {remote} [{parts[3]}]")
            self.samples += 1
            time.sleep(0.5)


def run_flow(report: Report) -> None:
    with isolated_app() as (client, api, schema_url):
        start = time.perf_counter()
        response = client.post(
            "/upload",
            files=[(
                "files",
                (
                    "offline-test.pdf",
                    make_pdf(
                        "A firewall monitors incoming and outgoing network traffic and blocks "
                        "packets that break its security rules. There are host-based and "
                        "network-based firewalls."
                    ),
                    "application/pdf",
                ),
            )],
        )
        report.check(response.status_code == 200, f"upload, parse, chunk, and embed ({time.perf_counter() - start:.1f} s)")
        doc_id = response.json()["documents"][0]["id"]

        def ask(question: str, conversation_id: str | None = None) -> dict:
            body = {"question": question, "doc_ids": [doc_id]}
            if conversation_id:
                body["conversation_id"] = conversation_id
            t = time.perf_counter()
            result = client.post("/ask", json=body)
            result.raise_for_status()
            data = result.json()
            data["seconds"] = time.perf_counter() - t
            return data

        # These check that each step works offline, not answer quality: the model's
        # wording varies between runs (it samples), so the answer is shown, not judged.
        first = ask("What does a firewall do?")
        report.check(
            bool(first["answer"].strip()),
            f"question answered by the local model ({first['seconds']:.1f} s, "
            f"{len(first['citations'])} citation(s)): {first['answer'][:80]!r}",
        )
        follow_up = ask("What types are there?", first["conversation_id"])
        report.check(
            bool(follow_up["answer"].strip()),
            f"follow-up question in the same chat ({follow_up['seconds']:.1f} s): {follow_up['answer'][:80]!r}",
        )
        cached = ask("What does a firewall do?")
        report.check(cached["cached"], f"repeat question served from cache ({cached['seconds']:.2f} s)")
        out_of_scope = ask("What is the capital of France?")
        report.check(
            out_of_scope["answer"] == "not found in your documents",
            f"out-of-scope question refused ({out_of_scope['seconds']:.1f} s)",
        )
        report.check(
            client.get(f"/conversations/{first['conversation_id']}/messages").status_code == 200,
            "chat history loads",
        )
        report.check(client.delete(f"/documents/{doc_id}").status_code == 200, "document deletion")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-network", action="store_true", help="run while online and report outbound connections")
    args = parser.parse_args()

    report = Report()
    mode = "online (connection audit)" if args.with_network else "offline"
    print(f"Offline check, mode: {mode}\n")

    reachable = internet_reachable()
    if args.with_network:
        report.note("internet reachable: " + ("; ".join(reachable) if reachable else "no"))
    else:
        if reachable:
            print("The internet is still reachable:\n  " + "\n  ".join(reachable))
            print("\nTurn off Wi-Fi / unplug Ethernet and run again, or use --with-network.")
            return 2
        report.check(True, "network is off (DNS lookups and TCP to public IPs fail)")

    watcher = ConnectionWatcher()
    watcher.start()
    try:
        run_flow(report)
    except Exception as exc:  # report, then fail
        report.check(False, f"flow raised {type(exc).__name__}: {exc}")
    finally:
        watcher.stop_event.set()
        watcher.join(timeout=5)

    data_handlers = sorted(c for c in watcher.remote if not c.lower().startswith(TRAY_APP))
    tray = sorted(c for c in watcher.remote if c.lower().startswith(TRAY_APP))
    report.check(
        not data_handlers,
        "no connections to non-local addresses by the app, the Ollama server, or the model runner "
        f"({watcher.samples} samples)" + ("" if not data_handlers else ": " + "; ".join(data_handlers)),
    )
    if tray:
        report.note("Ollama tray app (update checks, no user data): " + "; ".join(tray))

    passed = sum(status == "PASS" for status, _ in report.lines)
    summary = f"{passed} passed, {report.failed} failed"
    print(f"\n{summary}")

    out_dir = ROOT / "storage" / "privacy_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_file = out_dir / f"offline_check_{'online' if args.with_network else 'offline'}_{stamp}.txt"
    out_file.write_text(
        f"Offline check {stamp}, mode: {mode}, python {sys.version.split()[0]}, {os.name}\n"
        + "\n".join(f"[{status}] {text}" for status, text in report.lines)
        + f"\n{summary}\n",
        encoding="utf-8",
    )
    print(f"Saved to {out_file.relative_to(ROOT)}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
