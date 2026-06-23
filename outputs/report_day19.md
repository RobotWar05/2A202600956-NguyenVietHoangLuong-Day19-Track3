# Report Day 19 - GraphRAG with Tech Company Corpus

## 1. Objective

Build a GraphRAG pipeline from the Tech Company Corpus and compare it with a Flat RAG baseline.

## 2. Architecture

```text
dataset/*.txt
-> load documents
-> chunk text
-> extract entity-relation triples
-> deduplicate triples
-> build NetworkX knowledge graph
-> retrieve with Flat RAG and GraphRAG
-> compare benchmark answers
```

## 3. Graph Summary

- Nodes: 7188
- Edges: 9634
- Visualization: `outputs/graph.png`
- GraphML: `outputs/graph.graphml`

## 4. Evaluation Summary

Winner count:

- GraphRAG: 16
- Flat RAG: 2
- Tie: 2

|   question_id | question                                                                                        | which_is_better   | reason                                                                                                   |
|--------------:|:------------------------------------------------------------------------------------------------|:------------------|:---------------------------------------------------------------------------------------------------------|
|             1 | How are electric vehicle adoption and charging infrastructure connected in the corpus?          | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             2 | Which policies or incentives are linked to electric vehicle market growth?                      | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             3 | What relationship does the corpus show between the United States and China in technology trade? | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             4 | Which companies or organizations are connected to battery supply chains?                        | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             5 | How do export controls affect semiconductor or technology companies?                            | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             6 | What evidence links consumer incentives to EV adoption?                                         | Flat RAG          | This question asks for direct textual evidence, so passage retrieval is the more suitable baseline.      |
|             7 | Which markets or regions are mentioned as important for electric vehicle growth?                | Tie               | Both systems provide useful but different context: Flat RAG gives passages, GraphRAG gives entity links. |
|             8 | What entities are connected to clean vehicle tax credits?                                       | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|             9 | How are automakers connected to electric vehicles and batteries?                                | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            10 | Which entities are associated with charging stations or charging infrastructure?                | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            11 | What does the corpus say about China and electric vehicle manufacturing?                        | Tie               | Both systems provide useful but different context: Flat RAG gives passages, GraphRAG gives entity links. |
|            12 | Which government bodies or policies are linked to technology restrictions?                      | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            13 | How are critical minerals connected to US-China trade tensions?                                 | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            14 | Which entities are connected to public health or emissions benefits?                            | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            15 | What links exist between battery technology and EV costs?                                       | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            16 | Which companies are described as competitors or market participants in EVs?                     | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            17 | How are tariffs or trade restrictions connected to supply chains?                               | GraphRAG          | GraphRAG exposes entity-relation paths, which is stronger for this multi-hop style question.             |
|            18 | What multi-hop path connects EV policy, market adoption, and infrastructure?                    | GraphRAG          | This question explicitly asks for graph structure or multi-hop paths.                                    |
|            19 | Which documents provide evidence about EV market slowdown or growth?                            | Flat RAG          | This question asks for direct textual evidence, so passage retrieval is the more suitable baseline.      |
|            20 | What are the strongest graph relationships found in the corpus?                                 | GraphRAG          | This question explicitly asks for graph structure or multi-hop paths.                                    |

## 5. Conclusion

Flat RAG is useful for retrieving text passages that are semantically close to a question.
GraphRAG is more useful when the question needs explicit entity-relation links and multi-hop reasoning.

This submission uses an offline rule-based extractor as a reproducible baseline. The next upgrade is to replace triple extraction and final answer generation with an LLM while keeping the same graph construction and evaluation flow.
