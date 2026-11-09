# Privacy

## Principle

Your documents, their text, their embeddings, and your questions stay on your machine. The only network use is an explicit model download.

## What is stored, and where

| Data | Location | Format | Deleted when |
|------|----------|--------|--------------|
| Original documents | Not copied; read from your folders | As-is | Never by the app |
| Extracted text and chunks | Local index store | TBD | Document is removed |
| Embeddings | Local index store | TBD | Document is removed |
| Keyword index | Local index store | TBD | Document is removed |
| Chat history (v1.1) | Local file | TBD | User deletes it |
| Models | Local models folder | Model files | User deletes them |

Storage path: TBD. The user can change it.

## What leaves the machine

- Nothing during normal use.
- Model download: only when the user runs the explicit download command. The command lists the URLs before it starts.
- No telemetry, crash reporting, or analytics.

## Offline test

Steps to verify the claim:

1. Index the eval corpus with the network connected.
2. Disable all network interfaces (or enable airplane mode).
3. Run the full eval set and a set of interactive queries.
4. Confirm the app completes without errors.
5. Optionally capture traffic with a firewall or packet capture and confirm zero outbound connections.

Record the date, the tool used, and the result below.

| Date | Method | Outbound connections | Result |
|------|--------|---------------------|--------|
| | | | |

## Deletion test

1. Add a document with a unique phrase.
2. Confirm a query returns it.
3. Remove the document.
4. Confirm the phrase is absent from the index files and from query results.
