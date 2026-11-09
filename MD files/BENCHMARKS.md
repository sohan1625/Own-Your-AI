# Benchmarks

## Test machine

- CPU: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (8 logical processors)
- RAM: 24 GB (23.69 GB physical RAM)
- GPU (if any): NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, Driver 617.14, CUDA 13.4)
- Disk: SSD (Windows-SSD)
- OS: Windows 11 (64-bit)
- Power mode: plugged in / on battery

## Corpus

- Documents:
- Total pages:
- Total words:

## Method

- Run each measurement 5 times, report the median and the range.
- Close other heavy applications before measuring.
- Use the same corpus and eval set for every run.

## Indexing

| Metric | Result |
|--------|--------|
| Time per 100 pages | TBD |
| Peak RAM during indexing | TBD |
| Index size on disk per 100 pages | TBD |

## Querying

| Metric | p50 | p95 |
|--------|-----|-----|
| Retrieval time | TBD | TBD |
| End-to-end answer time | TBD | TBD |

Peak RAM during querying: TBD

## Quality vs. speed

| Model | Quantization | Recall@5 | Faithfulness | Answer time (p50) | Peak RAM |
|-------|-------------|----------|--------------|-------------------|----------|
| TBD | Full precision | | | | |
| TBD | Quantized (4-bit) | | | | |
