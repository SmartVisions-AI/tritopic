# Changelog

## 2.5.0 (October 2026)

Codebook mode and a research toolkit: start from the topics you expect, and get the reliability,
saturation, group-comparison and reporting outputs a paper needs.

### New

- **Seeded topics (codebook mode)**: `fit(documents, seeds={"name": "description" | [words]})`. The
  documents that match a seed best are pinned to one Leiden community per seed (`is_membership_fixed`);
  all other documents join a seeded topic or form emergent ones. `TopicInfo.seed`, `seed_topics_`,
  `emergent_topics_`, `seed_anchors_`, a `Seed` column in `get_topic_info()`; `seed_embeddings` for
  pre-computed embeddings; config `seed_anchors`, `seed_keyword_weight`. Full codebook on the dev splits:
  NMI 0.625 → 0.661, ARI 0.515 → 0.624; held-out ARI 0.399 → 0.498.
- **`tritopic.research`**:
  - `topic_reliability()`: bootstrap refits (default) or consensus runs; reliability and core share per
    topic, `document_stability_` per document. Reliable topics (≥ 0.7) were 77% pure vs. 54% below 0.5.
  - `saturation_curve()` / `plot_saturation()`: exact topic accumulation curve (hypergeometric), saturation
    points for 95% and 100% of the topics, optional refit method.
  - `bridge_documents()`, `topic_connections()`: documents and links between topics.
  - `topic_prevalence()` with Wilson or bootstrap intervals; `compare_groups()` with χ²/Fisher tests,
    Cramér's V, odds ratios and Benjamini-Hochberg correction; `distinctive_keywords()` (log-odds with
    informative prior).
  - `topic_evolution()` / `plot_evolution()`: per-period topics linked into births, splits, merges, deaths.
  - `topic_quotes()`: citable sentences per topic.
  - `methods_report()`: methods paragraph and parameter table from the fitted model.
- **`TopicInterpreter.codebook()`**: codebook entries for qualitative content analysis (definition,
  inclusion/exclusion criteria, coding notes) with real anchor quotes.
- **`intercoder_reliability()`** (Decisions API): an LLM as independent second coder; Cohen's kappa,
  per-topic precision/recall/F1, confusion matrix. BBC demo: kappa 0.94.
- Examples `09_seeded_topics.py`, `10_research_toolkit.py`, `11_codebook_and_second_coder.py`; user guide
  `docs/user_guide.md`; BBC demo extended with all new outputs.

### Changed

- `transform()` outlier threshold is calibrated on the training data by default
  (`outlier_threshold=None` → 1st percentile of the training similarities, `model.outlier_threshold_`).
  The fixed 0.35 rejected up to 40% of in-domain documents; the calibrated rule about 1%.

### Fixed

- `fit(verbose=True)` with automatic resolution crashed when printing the scan summary.

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
