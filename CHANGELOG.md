# Changelog

## 2.4.0 (October 2026)

Head-to-head with BERTopic on identical embeddings: NMI 0.582 vs. 0.439, ARI 0.440 vs. 0.285, keyword NPMI
0.262 vs. 0.117, 0% vs. 18% outliers (4 datasets × 5 topic counts × 3 seeds). See the README benchmarks.

### New

- **`TopicInterpreter`** (`tritopic.labeling`): an LLM reads each topic (keywords, proportionally sampled
  example documents, neighbouring topics) and returns label, description, aspects and a verdict
  (coherent / mixed / unclear) with sub-themes. `refine()` splits mixed topics when the graph split agrees
  with the LLM's sub-themes; `summarize()` writes an overview. OpenAI Responses API with a strict JSON
  schema, default model `gpt-6-luna`.
- **OpenAI Decisions API integration** (`tritopic.integrations.decisions`, optional): LLM word-intrusion
  test and topic ratings, `assign_documents()`, `reduce_outliers(strategy="decisions")`, merge suggestions.
- **Automatic topic count**: `n_topics="auto"` picks the Leiden resolution by keyword coherence, skipping
  partitions dominated by one topic (`auto_resolution`, `resolution_range`, `auto_resolution_steps`,
  `auto_resolution_tolerance`, `auto_resolution_max_share`; results in `resolution_`, `resolution_search_`).
- **Coverage-weighted c-TF-IDF**: `IDF · √(tf_share · coverage)`, +45% NPMI on the dev splits.
- Benchmark harness `benchmarks/compare_bertopic.py`, `llm_eval.py`, `summarize.py`,
  `results_to_markdown.py`; decision log in `benchmarks/experiments/`.
- BBC demo page in `examples/bbc_demo/`.

### Performance (fit 4.5x faster than 2.3 at equal NMI)

- Iterative refinement works on the reduced embeddings instead of calling `UMAP.transform()` each iteration.
- Consensus clustering on graph edges (O(edges)) instead of an all-pairs co-occurrence matrix.
- The corpus is tokenized once; the lexical TF-IDF view is derived from the keyword counts
  (arXiv, 55k characters per document: 44 s → 22 s).
- Batched keyword extraction; `build_hierarchy()` and `divide()` no longer re-tokenize per topic;
  vectorized coherence and `topic_overlap_matrix()`; lexical kNN graph cached.

### Fixes

- kNN graph on UMAP output used cosine distance; now Euclidean with a self-tuning kernel (`reduced_metric`).
  FAISS L2 distances are square-rooted.
- BM25 keywords scored the first *n* corpus documents instead of the topic's documents.
- Metadata connected every pair of documents sharing a category (millions of edges); it now reweights
  existing edges. String columns no longer fail with pandas 3.
- `reduce_topics()` applied a size penalty that merged small topics into large unrelated ones
  (2k → k: NMI 0.542 → 0.591); available as `size_penalty=0.3`.
- `n_topics=k` and `divide()` counted clusters below `min_cluster_size` and searched a fixed range; the
  search now widens until *k* is reachable and searches both directions.
- NPMI coherence used only each topic's own documents as reference and skipped bigram keywords; it now uses
  the whole corpus. Values are not comparable with earlier versions.
- Tokens must start with two letters (no `000`, `__`, years as keywords).
- `reduce_outliers()`, `reduce_topics()` and representative documents use the same unrefined embedding
  space as the topic centroids.

### Changed defaults

- `resolution` 1.0 → 0.3 (used for `n_topics=k` and `auto_resolution=False`).
- `n_topics="auto"` uses the coherence-based resolution search.

## 2.3.0 (February 2026)

Cross-lingual stop words (`language=`), automatic `BAAI/bge-m3` for `language="multilingual"`, hierarchical
topics (`build_hierarchy()`, `divide()`, `visualize_hierarchy_tree()`), per-document topics
(`get_document_topics()`, `topic_overlap_matrix()`, `visualize_overlap()`), graph-based soft assignment,
turftopic adapter.

## 2.2.x

Dimensionality reduction before graph building, soft topic probabilities, `reduce_outliers()`,
`reduce_topics()`, `merge_topics()`, save/load of the full model state.
