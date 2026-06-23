# Huong dan nop bai Day 19

## Trang thai hien tai

Bai da co du artifact toi thieu de nop:

- Code: `lab_day19.py`
- Bao cao chinh: `BAO_CAO_DAY19.md`
- Huong dan chay: `README.md`
- Thu vien: `requirements.txt`
- Anh graph: `outputs/graph.png`
- Bang benchmark 20 cau: `outputs/benchmark_results.csv`
- Bao cao chi phi: `outputs/cost_report.md`
- Report ngan: `outputs/report_day19.md`

## Can nop nhung gi?

Nen nop ca thu muc:

```text
2A202600956-NguyenVietHoangLuong-Day19-Track3/
```

Trong do bat buoc giu:

```text
lab_day19.py
BAO_CAO_DAY19.md
README.md
requirements.txt
outputs/graph.png
outputs/benchmark_results.csv
outputs/cost_report.md
outputs/report_day19.md
```

Co the giu them:

```text
outputs/chunks.jsonl
outputs/triples.jsonl
outputs/triples_clean.jsonl
outputs/graph.graphml
```

Neu file qua nang khi upload, co the bo 3 file JSONL lon va chi giu code + report + CSV + PNG.

## Lenh chay lai

PowerShell:

```powershell
cd E:\vin_ai_k2_2026_DocLap\Documents\Day19\2A202600956-NguyenVietHoangLuong-Day19-Track3
py -3 -m pip install -r requirements.txt
py -3 lab_day19.py
```

## Diem can noi neu bi hoi

Day la baseline GraphRAG offline:

- Khong dung API key.
- Triple extraction dang la rule-based.
- Uu diem: chay lai duoc, re, co graph va benchmark ro rang.
- Nhuoc diem: chat luong entity/relation khong bang LLM extractor.

Neu giao vien yeu cau GraphRAG dung LLM that, can nang cap ham `extract_triples_from_chunk()`.
