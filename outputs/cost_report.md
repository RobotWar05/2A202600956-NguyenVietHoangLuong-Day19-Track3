# Cost Report - Lab Day 19

This run used the offline rule-based extractor, so API token cost is 0.
Estimated tokens below are approximate input tokens if the same chunks were sent to an LLM.

## Summary

- Documents processed: 70
- Chunks created: 826
- Raw triples: 17062
- Clean triples after deduplication: 10387
- Graph nodes: 7188
- Graph edges: 9634
- Indexing/runtime seconds: 19.68
- Approximate input characters: 3901285
- Approximate input tokens: 975322
- Actual API cost in this run: 0 USD

## Relation Counts

- RELATED_TO: 3966
- LOCATED_IN: 2217
- HAS_MARKET: 719
- USES_TECHNOLOGY: 642
- DEVELOPS: 566
- AFFECTED_BY: 537
- REGULATED_BY: 480
- INVESTS_IN: 415
- COMPETES_WITH: 222
- SUPPLIES_TO: 172
- RESTRICTS: 146
- PARTNERS_WITH: 124
- FOUNDED_IN: 115
- ACQUIRES: 48
- FOUNDED_BY: 18

## Top Graph Nodes

- China: degree 1107
- battery: degree 277
- electric vehicle: degree 227
- Tesla: degree 225
- Biden: degree 208
- United States: degree 164
- batteries: degree 140
- BYD: degree 133
- Nio: degree 124
- Ford: degree 107
- Europe: degree 96
- Polestar: degree 88
- electric vehicles: degree 88
- COVID-19: degree 85
- USD: degree 78
- tariffs: degree 74
- supply chain: degree 67
- Chinese EV: degree 64
- semiconductor: degree 59
- charging infrastructure: degree 58

## Technical Notes

- This is a baseline GraphRAG pipeline.
- Rule-based extraction is deterministic and cheap, but lower quality than LLM extraction.
- For final quality improvement, replace the extractor with an LLM JSON triple extractor and keep the same downstream graph/evaluation code.