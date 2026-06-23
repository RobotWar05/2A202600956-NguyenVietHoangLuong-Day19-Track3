# Bao cao Lab Day 19 - GraphRAG voi Tech Company Corpus

## 1. Muc tieu

Bai lab nay xay dung mot pipeline GraphRAG cho bo du lieu Tech Company Corpus.
Muc tieu la so sanh cach tra loi cua Flat RAG va GraphRAG tren cac cau hoi can lien ket nhieu thuc the.

Ket qua nop bai gom:

- Ma nguon: `lab_day19.py`
- Anh knowledge graph: `outputs/graph.png`
- Bang so sanh 20 cau hoi: `outputs/benchmark_results.csv`
- Bao cao chi phi: `outputs/cost_report.md`

## 2. Kien truc he thong

Pipeline duoc chia thanh cac buoc:

```text
dataset/*.txt
-> load documents
-> chunk text
-> extract entity-relation triples
-> deduplicate triples
-> build NetworkX knowledge graph
-> Flat RAG retrieval
-> GraphRAG traversal
-> benchmark and cost report
```

Ly do chon NetworkX:

- Chay offline, khong can database ngoai.
- Phu hop prototype nhanh.
- De ve graph bang Matplotlib.
- De kiem tra node, edge, degree va subgraph.

## 3. Xu ly du lieu

Bo du lieu gom 70 file text trong thu muc `dataset`.
Do co file rat lon, pipeline khong day ca file vao mot lan ma chia chunk truoc khi trich xuat thong tin.

Thong ke lan chay cuoi:

- Documents processed: 70
- Chunks created: 826
- Raw triples: 17062
- Clean triples after deduplication: 10387
- Graph nodes: 7188
- Graph edges: 9634

## 4. Entity Extraction va Relation Extraction

Trong ban nop nay, phan trich xuat triples su dung rule-based extractor de dam bao co the chay lai offline, khong phu thuoc API key.

Moi triple co dang:

```text
(subject, relation, object)
```

Vi du relation:

- `AFFECTED_BY`
- `REGULATED_BY`
- `INVESTS_IN`
- `SUPPLIES_TO`
- `RESTRICTS`
- `USES_TECHNOLOGY`
- `HAS_MARKET`

Han che cua ban rule-based:

- Do chinh xac entity/relation khong bang LLM extractor.
- Co the con mot so entity nhieu/noise hoac quan he chung chung.

Uu diem:

- Chay nhanh.
- Khong ton chi phi API.
- Reproducible, nguoi cham co the chay lai.

## 5. Graph Construction va Deduplication

Pipeline co buoc deduplicate triple truoc khi build graph.

Deduplication quan trong vi cung mot thuc the co the xuat hien duoi nhieu ten:

```text
US
U.S.
United States
```

Neu khong gop lai, graph se bi tach thanh nhieu node nho, lam multi-hop traversal sai hoac thieu thong tin.

Graph duoc luu va truc quan hoa:

- `outputs/graph.png`
- `outputs/graph.graphml` (file trung gian, co the khong nop neu upload bi gioi han)

## 6. Query Answering

Flat RAG:

- Dung TF-IDF retrieval de tim chunk gan cau hoi nhat.
- Phu hop cau hoi can bang chung truc tiep trong van ban.

GraphRAG:

- Tim entity lien quan trong cau hoi.
- Duyet graph theo cac node lan can.
- Tra ve cac duong lien ket entity-relation.
- Phu hop cau hoi multi-hop.

Khac biet cot loi:

- Flat RAG tra loi dua tren do tuong dong van ban.
- GraphRAG tra loi dua tren cau truc quan he giua thuc the.

## 7. Benchmark

Benchmark gom 20 cau hoi.
Ket qua duoc luu tai:

```text
outputs/benchmark_results.csv
```

Ket qua tong hop:

- GraphRAG tot hon: 16 cau
- Flat RAG tot hon: 2 cau
- Hoa: 2 cau

Nhan xet:

- GraphRAG tot hon voi cau hoi can lien ket chinh sach, cong ty, quoc gia, cong nghe va chuoi cung ung.
- Flat RAG tot hon voi cau hoi yeu cau trich bang chung truc tiep tu document.
- Ket qua nay phu hop voi ban chat cua hai phuong phap.

## 8. Cost Analysis

Lan chay hien tai khong dung API nen chi phi API la 0 USD.

Thong ke uoc tinh:

- Approximate input characters: 3901285
- Approximate input tokens: 975322
- Actual API cost: 0 USD
- Runtime: xem `outputs/cost_report.md`

Neu thay rule-based extractor bang LLM extractor, chi phi se phu thuoc vao model va tong token cua cac chunk.

## 9. Ket luan

Pipeline da hoan thanh cac yeu cau chinh cua Lab Day 19:

- Doc va chunk corpus.
- Trich xuat triples.
- Deduplicate entity/relation.
- Build knowledge graph.
- Truc quan hoa graph.
- So sanh Flat RAG va GraphRAG tren 20 cau hoi.
- Bao cao chi phi indexing.

Huong cai thien tiep theo la thay rule-based extractor bang LLM JSON extractor de tang chat luong triples.
