# 2A202600956-NguyenVietHoangLuong-Day19-Track3

Lab Day 19: Build a GraphRAG pipeline with the Tech Company Corpus.

## What this project does

This project provides an offline baseline GraphRAG pipeline:

```text
dataset/*.txt
-> load documents
-> chunk text
-> extract entity-relation triples
-> deduplicate triples
-> build NetworkX graph
-> draw graph image
-> compare Flat RAG vs GraphRAG on 20 benchmark questions
```

The current extractor is rule-based, so it does not require an API key. It is good enough for a reproducible prototype and report. For higher quality, replace the extraction function with an LLM JSON triple extractor.

## Setup

Run in PowerShell:

```powershell
cd E:\vin_ai_k2_2026_DocLap\Documents\Day19\2A202600956-NguyenVietHoangLuong-Day19-Track3
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the packages already exist on your machine, you can run directly without creating `.venv`.

## Quick test with 5 documents

```powershell
python lab_day19.py --max-docs 5
```

## Full run with all 70 documents

```powershell
python lab_day19.py
```

## Outputs

After running, check the `outputs/` folder:

- `chunks.jsonl`: chunked dataset
- `triples.jsonl`: raw extracted triples
- `triples_clean.jsonl`: deduplicated triples
- `graph.graphml`: graph file
- `graph.png`: graph screenshot for submission
- `benchmark_results.csv`: 20-question Flat RAG vs GraphRAG comparison
- `cost_report.md`: runtime/token/cost report
- `report_day19.md`: short final report

Large intermediate files are generated for audit/debug but ignored by Git:

- `chunks.jsonl`
- `triples.jsonl`
- `triples_clean.jsonl`
- `graph.graphml`

## Submission checklist

- Source code: `lab_day19.py`
- Main report: `BAO_CAO_DAY19.md`
- Graph screenshot: `outputs/graph.png`
- Benchmark table: `outputs/benchmark_results.csv`
- Cost analysis: `outputs/cost_report.md`
- Report: `outputs/report_day19.md`
