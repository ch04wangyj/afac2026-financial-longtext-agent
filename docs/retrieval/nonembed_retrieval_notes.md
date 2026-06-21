# Non-Embedding Retrieval Design Notes

## Scope

These notes summarize a few lexical / non-embedding retrieval directions relevant to the AFAC2026 financial long-document benchmark.

The goal is not to reproduce paper systems exactly, but to extract design constraints that help improve sparse retrieval without query expansion or dense embeddings.

---

## 1. Overcoming low-utility facets for complex answer retrieval
- **Paper:** `1811.08772v1`
- **Title:** *Overcoming low-utility facets for complex answer retrieval*

### Takeaways
1. Complex queries contain facets of very different utility; some lexical facets help, others add noise.
2. A retrieval system should not assume every query fragment deserves equal downstream reward.
3. Structuring retrieval stages around more useful facets can outperform naïve lexical accumulation.

### What we can borrow here
- Separate document identity facets from answer-bearing metric facets.
- Avoid simplistic “more query terms matched = always better” logic.
- Prefer staged retrieval where the first stage gets the right report family and the second stage finds the right answer-bearing block.

### What we should not implement
- Topic-specific facet engineering that only works for this single benchmark.
- Learned neural facet models.

---

## 2. Match Your Words! A Study of Lexical Matching in Neural Information Retrieval
- **Paper:** `2112.05662v2`
- **Title:** *Match Your Words! A Study of Lexical Matching in Neural Information Retrieval*

### Takeaways
1. Strong lexical matching remains fundamental, especially under distribution shift.
2. Out-of-domain retrieval often still depends heavily on exact or near-exact term alignment.
3. Replacing lexical matching with more complicated heuristics can hurt robustness if core lexical grounding is weakened.

### What we can borrow here
- Keep BM25 / lexical recall as the backbone.
- Improve stage ordering and structure-aware reranking rather than abandoning lexical retrieval.
- Treat document-level and chunk-level lexical retrieval as first-class components.

### What we should not implement
- Neural matching models or embedding rerankers.
- Heavy learned sparse models requiring new training data.

---

## 3. On the Interpolation of Contextualized Term-based Ranking with BM25 for Query-by-Example Retrieval
- **Paper:** `2210.05512v1`
- **Title:** *On the Interpolation of Contextualized Term-based Ranking with BM25 for Query-by-Example Retrieval*

### Takeaways
1. BM25 remains a strong backbone even when combined with richer term-based signals.
2. Interpolation of complementary signals is often safer than replacing BM25 with a stronger but narrower scorer.
3. Keeping a robust first-stage lexical retriever helps preserve recall.

### What we can borrow here
- Interpolate complementary **non-neural** signals with BM25 rather than replacing it.
- In this repo, that means: document BM25 shortlist + chunk BM25/BM25F-lite + structure-only rerank.
- Preserve broad recall first; add more specialized ranking logic second.

### What we should not implement
- Contextualized transformer term models.
- Query-expansion pipelines that introduce new semantic terms.

---

## 4. Design rules for this repo

### Allowed
- Document-level BM25 shortlist using `DocumentSearchIndex`.
- Chunk-level BM25 and BM25F-lite fusion inside shortlisted docs.
- Structure-only rerank using:
  - statement/table anchors,
  - accounting-section markers,
  - numeric density,
  - same-page/same-block evidence grouping,
  - generic distractor penalties.

### Disallowed
- Direct rerank bonus based on the number of benchmark bundle terms appearing in a chunk.
- Synonym expansion from `归母净利润` to longer canonical phrases inside retrieval.
- Dense embedding or neural rerankers.

---

## 5. Concrete implication for the current BYD benchmark

The currently passing `doc_first_chunk_rerank` is not acceptable as final because it directly rewards benchmark bundle term hits.

The next acceptable design should look like:

```text
DocumentSearchIndex shortlist
-> chunk BM25/BM25F-lite retrieval inside shortlist
-> fusion
-> structure-only rerank
-> optional same-page / statement-pack promotion
```

This keeps the retrieval lexical, non-embedding, and more generalizable across financial-report questions.
