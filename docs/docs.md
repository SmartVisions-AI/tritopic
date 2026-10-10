# TriTopic 2.5 — Technical Documentation

How TriTopic works, why it is built this way, and how to use every part of it. For a quick start see the
[README](../README.md), for task-by-task code the [user guide](user_guide.md); for release notes the [CHANGELOG](../CHANGELOG.md).

## Contents

1. [Design principles](#1-design-principles)
2. [The pipeline step by step](#2-the-pipeline-step-by-step)
3. [Graph construction](#3-graph-construction)
4. [Consensus clustering](#4-consensus-clustering)
5. [Choosing the resolution](#5-choosing-the-resolution)
6. [Iterative refinement](#6-iterative-refinement)
7. [Keywords](#7-keywords)
8. [Centroids, probabilities and new documents](#8-centroids-probabilities-and-new-documents)
9. [Post-fit operations](#9-post-fit-operations)
10. [Hierarchies and per-document analysis](#10-hierarchies-and-per-document-analysis)
11. [Languages](#11-languages)
12. [LLM topic interpretation](#12-llm-topic-interpretation)
13. [LLM judgements via the Decisions API](#13-llm-judgements-via-the-decisions-api)
14. [Evaluation metrics](#14-evaluation-metrics)
15. [Configuration reference](#15-configuration-reference)
16. [Design decisions and the evidence behind them](#16-design-decisions-and-the-evidence-behind-them)
17. [Performance](#17-performance)
18. [Seeded topics (codebook mode)](#18-seeded-topics-codebook-mode)
19. [Research toolkit](#19-research-toolkit)

---

## 1. Design principles

1. **No single view is enough.** Embeddings capture meaning but blur specific wording; TF-IDF captures
   wording but misses synonyms; metadata captures structure but not content. TriTopic fuses them in one
   graph. Removing the lexical view costs about 0.05 NMI on the development benchmarks.
2. **Stochastic clustering needs stabilising.** One Leiden run depends on its seed. TriTopic runs it ten
   times and builds a consensus; across seeds the NMI spread is 0.014 (BERTopic: 0.105).
3. **Every document gets a topic.** Graph clustering assigns every document; only clusters smaller than
   `min_cluster_size` become outliers, which is rare.
4. **Decide on development data, report on held-out data.** Every default in 2.4 was chosen on separate
   development splits; the benchmark numbers come from untouched evaluation splits.

## 2. The pipeline step by step

`model.fit(documents, embeddings=None, metadata=None)`:

| Step | What happens | Stored as |
|---|---|---|
| 1 | Encode documents with a sentence-transformer (skipped if `embeddings` are passed) | `original_embeddings_` |
| 2 | Reduce to 10 dimensions with UMAP (`min_dist=0`, cosine input) | `reduced_embeddings_`, `_dim_reducer` |
| 3 | Tokenize once; derive the TF-IDF matrix from the counts | `lexical_matrix_` |
| 4 | Encode metadata (optional) | — |
| 5 | Build the fused graph; with `n_topics="auto"` scan resolutions | `graph_`, `resolution_`, `resolution_search_` |
| 6 | Consensus Leiden + iterative refinement | `labels_` |
| 7 | Keywords and representative documents per topic | `topics_` |
| 8 | Centroids and soft probabilities | `topic_embeddings_`, `probabilities_` |
| 9 | With `n_topics=k`: resolution search for *k* topics, merge down if needed | — |

Full-dimensional, unrefined embeddings are used for centroids, probabilities, `transform()`, outlier
reduction and merging, so new documents are always compared in the same space.

## 3. Graph construction

**Semantic graph** (`graph_type="hybrid"`, default). kNN on the reduced embeddings with Euclidean distance
(`reduced_metric`). UMAP output is a Euclidean layout; cosine angles around its arbitrary origin would join
clusters that happen to lie on the same ray. Similarities use a self-tuning Gaussian kernel,
`exp(-d² / (σᵢ σⱼ))` with σ the distance to the k-th neighbour, so dense and sparse regions get comparable
weights. Without dimensionality reduction the metric is cosine on the full embeddings.

- *Mutual kNN*: an edge only if both points are among each other's neighbours (removes noise bridges).
- *SNN*: weight = shared neighbours / k (structural similarity), computed as a sparse product `A·Aᵀ`.
- *Hybrid*: `(1 - snn_weight) · mutual + snn_weight · SNN`, each max-normalised.

**Lexical graph.** Mutual kNN with cosine similarity on TF-IDF (top 10,000 terms, sublinear TF, uni- and
bigrams, stop words of `language`). It is built once per fit and cached.

**Fusion.** `w_sem · semantic + w_lex · lexical + 0.1 · min(semantic, lexical)`; weights are renormalised
over the active views. The last term rewards edges both views agree on.

**Metadata.** Columns are encoded per document (categorical: exact match, including strings and booleans;
numerical and datetime: min-max scaled, similar if `1 - |Δ| > 0.8`). The metadata similarity of an edge,
averaged over columns, is added with weight `metadata_weight` to **existing** edges only. Connecting every
pair of documents that share a category would add O(n²) edges and let metadata dominate the topics.

## 4. Consensus clustering

1. Leiden (`RBConfigurationVertexPartition`, weighted) runs `n_consensus_runs` times with seeds
   `random_state + i`.
2. For every graph edge, the **agreement** is the share of runs that put both ends in the same cluster
   (Lancichinetti & Fortunato, 2012).
3. Edges with agreement below 50% are dropped (each node keeps its most consistent edge), the rest are
   weighted by `agreement × original weight`, and Leiden runs again on this consensus graph.
4. Steps 2-3 repeat until all runs agree or the consensus graph stops changing (max. 5 rounds); the run
   with the highest mean ARI to the others is returned.
5. Clusters smaller than `min_cluster_size` become outliers (-1).

Memory and time grow with the number of edges. Earlier versions built an all-pairs co-occurrence matrix,
which needed several GB beyond ~20k documents. `stability_score_` is the mean pairwise ARI of the initial
runs.

## 5. Choosing the resolution

The Leiden resolution sets the granularity (higher = more topics).

**`n_topics="auto"`** (with `auto_resolution=True`, default): on the initial graph, one Leiden run per
resolution in `resolution_range` (default 0.01-1.0, 15 log-spaced steps). For each partition, keywords are
extracted and scored by mean NPMI from the cached document-term matrix. TriTopic keeps the **coarsest**
resolution whose coherence is within `auto_resolution_tolerance` (5%) of the best, among partitions whose
largest topic holds at most `auto_resolution_max_share` (50%) of the documents. Very coarse partitions score
high NPMI on generic, frequently co-occurring words; the share guard prevents collapses to two topics.

On the development splits (3 seeds) this gave NMI 0.594 / ARI 0.474, against 0.565 / 0.415 for a fixed
resolution. It still tends to find more topics than a corpus has classes.

**`n_topics=k`**: log-space bisection for a resolution with *k* clusters of at least `min_cluster_size`.
The bracket widens (×4 steps) while the count still moves towards *k*; on very clean or very noisy graphs a
fixed range silently missed the target. If the result overshoots, `reduce_topics(k)` merges down.

**Fixed**: `auto_resolution=False` uses `resolution` (default 0.3; Leiden's usual 1.0 over-segments kNN
graphs badly, e.g. 60 topics for 6 newsgroup categories).

## 6. Iterative refinement

After clustering, the graph-building embeddings (the reduced ones) are pulled towards their topic centroid
and the semantic graph is re-clustered:

```
refined = (1 - b) · x + b · centroid,    b = blend · 1 / (1 + (d / median_d)²)
```

`blend` decays from 0.3 to 0.1 over `max_iterations`; documents far from the centroid move less. The loop
stops when the ARI between consecutive partitions reaches `convergence_threshold` (0.95). Refinement works
in the reduced space directly; earlier versions re-projected the full embeddings through
`UMAP.transform()` every iteration, which was slow and noisy. On the development benchmarks refinement does
not hurt but has no clear effect either; it can be switched off with `use_iterative_refinement=False`.

## 7. Keywords

The corpus is tokenized once (`CountVectorizer`, uni- and bigrams, stop words of `language`,
`min_df=2`, `max_df=0.95`). Tokens must start with two letters, so `000`, `__` or years never become
keywords. The same counts feed the lexical view.

**Coverage-weighted c-TF-IDF** (`keyword_method="ctfidf"`, default):

```
score(t) = IDF(t) · √( tf_share(t) · coverage(t) )
tf_share(t) = share of the topic's tokens that are t
coverage(t) = share of the topic's documents that contain t
IDF(t)      = log(N / (1 + df(t)))
```

Term frequency alone favours words repeated in a few long documents; coverage alone favours boilerplate
("thanks", "mail address"). The geometric mean raised NPMI coherence from 0.210 to 0.305 on the development
splits at equal diversity. BERTopic's formula (`tf · log(1 + A/f)`) scored 0.193 in the same test.

**BM25** (`"bm25"`): average Okapi BM25 weight of a term in the topic's documents relative to the corpus
average, times `log(1 + frequency)`. **KeyBERT** (`"keybert"`): embedding-based candidates with MMR.

All topics are scored in one sparse product; `build_hierarchy()` and `divide()` reuse the cached matrix.

## 8. Centroids, probabilities and new documents

- `topic_embeddings_`: mean of the unrefined embeddings per topic.
- `probabilities_` (`soft_assignment_method="centroid"`): `softmax(cos(x, centroids) · softmax_temperature)`
  (temperature 5). With `"graph"`, the topic distribution of a document's weighted graph neighbours.
- `transform(docs)`: encode, nearest centroid by cosine; below the outlier threshold → -1. By default
  (`outlier_threshold=None`) the threshold is calibrated on the training data: the 1st percentile of the
  training documents' similarity to their own centroid (`model.outlier_threshold_`). The fixed 0.35 used up
  to 2.4 rejected up to 40% of in-domain documents on the benchmark corpora, the calibrated rule about 1%.
  `transform_proba(docs)` returns the softmax distribution.

## 9. Post-fit operations

All operations refresh keywords, centroids and probabilities and reuse the cached token counts.

- **`reduce_outliers(strategy)`**: `"embeddings"` (nearest centroid above a threshold), `"neighbors"`
  (majority vote of the nearest non-outlier documents), `"decisions"` (an LLM picks the topic or "other",
  see §13).
- **`reduce_topics(n, size_penalty=0.0)`**: repeatedly merges the two topics with the most similar
  centroids. A size penalty `(min/max size)^p` is optional; the 0.3 used up to 2.4.0 merged small topics
  into large unrelated ones (reducing 2k → k topics: NMI 0.542 without vs. 0.591 with the fix on the
  evaluation splits). Graph-based agglomeration by modularity gain was tested and is worse.
- **`merge_topics([ids])`**: merges the given topics into the largest one.
- **`divide(topic_id, n_subtopics)`**: re-clusters the topic's subgraph with a resolution search in both
  directions; the new topics get fresh ids.

## 10. Hierarchies and per-document analysis

**`build_hierarchy(n_levels=3)`** clusters the graph at resolutions spaced around `resolution_`
(`/4 … ×4`) and links each fine node to the coarse node holding most of its documents.
`TopicHierarchy.cut(depth)`, `flatten()`, `get_node(id)`; `visualize_hierarchy_tree()` draws it.

**`get_document_topics(doc_idx, top_n)`** returns the top topics of one document (centroid or graph method).
**`topic_overlap_matrix(threshold)`** counts how often two topics are both "active" (probability above the
threshold) in the same document; `visualize_overlap()` shows it as a heatmap.

## 11. Languages

`language` sets the stop words for keywords, the lexical view and coherence: `"english"` (scikit-learn
list), `"german"`, `"french"`, `"spanish"` (built-in lists) or `"multilingual"` (no stop words). With
`"multilingual"` and the default embedding model, TriTopic switches to `BAAI/bge-m3`. The token pattern
accepts letters of any alphabet.

## 12. LLM topic interpretation

`tritopic.TopicInterpreter` (OpenAI Responses API with a strict JSON schema; default model `gpt-6-luna`;
needs `OPENAI_API_KEY`, no extra package).

**Evidence per topic.** The top 15 keywords; `n_docs` (8) example documents chosen by k-means on the topic's
embeddings (the document nearest each cluster centre, largest cluster first), so a large sub-group gets
several examples while the topic's spread is still covered; the three nearest topics for contrast; topic
size and share; an optional `domain_hint`.

**Answer** (`TopicInterpretation`): `label`, `description`, `aspects`, `verdict` (`coherent`, `mixed`,
`unclear`), `sub_themes` (name, description, supporting example documents), `confidence`, `evidence`.
`interpret(model)` writes labels and descriptions into `model.topics_` and stores all results in
`model.interpretations_`.

**`refine(model)`** splits a topic with `divide()` when

1. the verdict is `mixed` with confidence ≥ 0.6,
2. at least two sub-themes are each backed by ≥ 2 example documents (one stray article does not make a
   topic mixed), and
3. the acceptance rule holds (`accept="consistency"`, default): the example documents of different
   sub-themes land in different new topics, each sub-theme's examples mostly in one. `"coherence"`
   (keyword NPMI of the new topics beats the original), `"both"` and `"none"` are available.

Rejected splits are undone; the new topics are interpreted. `summarize(model)` returns a short overview and
a grouping of related topics (`model.overview_`).

Acceptance rules compared on identical LLM verdicts (10 dataset settings × 2 interpretation runs):

| Rule | Mean NMI gain | Worst case |
|---|---|---|
| none (keep every split) | +0.044 | −0.017 |
| **consistency** (default) | +0.038 | −0.001 |
| coherence | +0.033 | −0.021 |
| both | +0.026 | −0.021 |

With the default, `refine()` raised NMI from 0.614 to 0.658 on the development splits and from 0.606 to
0.639 on the held-out evaluation splits (BBC with 3 topics: 0.716 → 0.859).

**Model choice** (development splits, deliberately coarse topics, 28 topics of which 15 genuinely mixed):

| Model | Mixed topics found | False alarms | refine(): NMI | Time per corpus |
|---|---|---|---|---|
| gpt-6-luna | 9 | 5 | 0.614 → 0.654 | 29 s |
| gpt-5.5 | 8 | 9 | 0.614 → 0.638 | 41 s |
| gpt-5.4-mini | 4 | 4 | 0.614 → 0.625 | 16 s |

Two lessons from the development: farthest-point sampling of examples over-represented spread-out
minorities (the BBC politics/business topic got one politics and six business examples), and keyword NPMI
is a poor referee for splits (business keywords are naturally less coherent than political ones, so the
correct split was rejected). Proportional sampling and the consistency check fixed both.

## 13. LLM judgements via the Decisions API

`tritopic.integrations.decisions` uses OpenAI's Decisions API (`POST /v1/decisions`, model `gpt-6-luna`,
typed answers: `predicate` → probability, `choice` → one option, `score` → ordered levels). `DecisionsClient`
runs requests in parallel, retries on 429/5xx and caches identical requests.

| Function | Question type | What it does |
|---|---|---|
| `word_intrusion(keywords, client)` | choice | Top-5 keywords + one top keyword of another topic; which word does not belong? Returns accuracy and mean probability on the intruder (Chang et al., 2009). |
| `rate_topics(keywords, client)` | score | 0 (unrelated) … 3 (one clear, nameable theme), optionally with example documents. |
| `assign_documents(model, docs, client)` | choice | Assign documents to topics (label + keywords + 2 example snippets; with many topics the 10 nearest centroids are the candidates). `allow_other=True` adds an abstain option. |
| `reduce_outliers(strategy="decisions")` | choice | As above with the abstain option; abstaining keeps the outlier. |
| `suggest_merges(model, client)`, `apply_merges` | predicate | Asks for the most similar topic pairs whether they describe the same theme. |
| `intercoder_reliability(model, client, sample_size=200)` | choice | The LLM codes a random sample like a second human coder (labels, descriptions, keywords only). Returns `IntercoderResult`: Cohen's kappa, agreement, per-topic precision/recall/F1, confusion matrix. |

Measured on the evaluation splits: LLM topic rating 2.21 for TriTopic vs. 1.94 (BERTopic tuned) and 0.49
(default); assignment accuracy 0.657 vs. 0.648 for nearest-centroid, about 74% on the documents it does not
abstain on; merge judgements separate same-class pairs better than centroid cosine (AUC 0.63-0.75 vs.
0.61-0.65) but are conservative, so merges rarely change NMI.

## 14. Evaluation metrics

`model.evaluate()`:

- `coherence_mean`, `coherence_std`: NPMI of each topic's keywords over the **whole corpus** (document
  co-occurrence; pairs that never co-occur count as -1; bigram keywords are matched with the keyword
  analyzer). Not comparable with versions before 2.4, which used only each topic's own documents.
- `diversity`: unique share of all topics' keywords.
- `stability`: mean pairwise ARI of the consensus runs.
- `n_topics`, `outlier_ratio`.

Standalone in `tritopic.utils.metrics`: `compute_coherence`, `compute_coherence_batch`,
`coherence_from_doc_term`, `compute_diversity`, `compute_stability`, `compute_silhouette`,
`compute_downstream_score`.

## 15. Configuration reference

| Parameter | Default | Meaning |
|---|---|---|
| `language` | `"english"` | Stop words; `"multilingual"` selects `BAAI/bge-m3` |
| `embedding_model` | `"all-MiniLM-L6-v2"` | sentence-transformers model |
| `embedding_batch_size` | 32 | Encoding batch size |
| `use_dim_reduction` | True | UMAP before graph building |
| `reduced_dims` | 10 | Target dimensions |
| `dim_reduction_method` | `"umap"` | or `"pacmap"` |
| `umap_n_neighbors`, `umap_min_dist` | 15, 0.0 | UMAP parameters |
| `reduced_metric` | `"euclidean"` | kNN metric on reduced embeddings |
| `metric` | `"cosine"` | kNN metric without reduction |
| `n_neighbors` | 15 | k of the kNN graphs |
| `graph_type` | `"hybrid"` | `"knn"`, `"mutual_knn"`, `"snn"`, `"hybrid"` |
| `snn_weight` | 0.5 | SNN share in the hybrid graph |
| `use_lexical_view`, `use_metadata_view` | True, False | Views |
| `semantic_weight`, `lexical_weight`, `metadata_weight` | 0.5, 0.3, 0.2 | View weights (renormalised) |
| `auto_resolution` | True | Coherence-based resolution for `n_topics="auto"` |
| `resolution_range` | None → (0.01, 1.0) | Scan range |
| `auto_resolution_steps` | 15 | Resolutions scanned |
| `auto_resolution_tolerance` | 0.05 | Within 5% of the best coherence |
| `auto_resolution_max_share` | 0.5 | Skip partitions dominated by one topic |
| `resolution` | 0.3 | Start of the `n_topics=k` search; used if `auto_resolution=False` |
| `n_consensus_runs` | 10 | Leiden runs per consensus |
| `min_cluster_size` | 5 | Smaller clusters become outliers |
| `use_iterative_refinement` | True | Refinement loop |
| `max_iterations`, `convergence_threshold` | 5, 0.95 | Loop limits |
| `n_keywords`, `n_representative_docs` | 10, 5 | Per topic |
| `keyword_method` | `"ctfidf"` | or `"bm25"`, `"keybert"` |
| `soft_assignment_method` | `"centroid"` | or `"graph"` |
| `softmax_temperature` | 5.0 | Sharpness of probabilities |
| `outlier_threshold` | None | `transform()` cut-off; None = calibrated (1st percentile of training similarities) |
| `seed_anchors` | None | Anchor documents per seed; None = 1% of the corpus, clipped to 5-30 |
| `seed_keyword_weight` | 0.5 | Weight of the seed words vs. semantic similarity when picking anchors |
| `seed_aware_resolution` | False | Anchored resolution scan (tested worse, off) |
| `random_state`, `verbose` | 42, True | |

## 16. Design decisions and the evidence behind them

| Decision | Evidence (development splits unless noted) |
|---|---|
| Leiden on a kNN graph instead of HDBSCAN | 0% vs. 18% outliers; NMI 0.582 vs. 0.439 (evaluation splits) |
| Consensus of 10 runs | NMI spread across seeds 0.014 vs. 0.105 for BERTopic |
| Lexical view on | Without it NMI −0.05 (ablation) |
| Graph variants, UMAP dims, k = 30 | Within seed noise of the defaults; defaults kept |
| Euclidean kernel on UMAP output | Small but consistent gain over cosine; cosine is geometrically wrong there |
| Coverage-weighted c-TF-IDF | NPMI 0.305 vs. 0.210 (old) and 0.193 (BERTopic formula) |
| Coherence-based auto resolution with share guard | NMI 0.594 vs. 0.565 (fixed 0.3); worst run 0.390 vs. 0.041 without the guard |
| No size penalty in `reduce_topics` | 2k → k: NMI 0.632 vs. 0.492 (dev), 0.591 vs. 0.542 (evaluation) |
| `gpt-6-luna` for interpretation | Most mixed topics found, best refine() gain |
| Seeds as fixed Leiden memberships | Full codebook: NMI 0.625 → 0.661, ARI 0.515 → 0.624 (dev); held-out ARI 0.399 → 0.498 |
| Seed-aware resolution scan off | The anchored scan chose worse partitions than the plain scan |
| Bootstrap reliability as default | Spearman rho with purity 0.42 vs. 0.26 for the consensus variant |
| Exact accumulation curve for saturation | Michaelis-Menten extrapolation was unstable (AG News: 69 estimated topics) |
| Calibrated `transform()` threshold | About 1% vs. up to 40% rejected in-domain documents |

All experiments are archived in [`benchmarks/experiments`](../benchmarks/experiments/README.md).

## 17. Performance

Fit times with pre-computed embeddings (one machine, sequential runs):

| Corpus | TriTopic 2.3 | TriTopic 2.4 | BERTopic tuned |
|---|---|---|---|
| 20 Newsgroups, 2,000 short posts | 35 s | 7 s | 6 s |
| arXiv, 2,000 papers (55k characters each) | 102 s | 23 s | 50 s |
| 20 Newsgroups, 17,900 posts | 148 s | 34-44 s | — |

The one-time UMAP fit dominates on short texts, tokenization on long ones. `TopicInterpreter` adds about
2-3 s per topic of API time (parallelised over 8 workers).

## 18. Seeded topics (codebook mode)

`fit(documents, seeds={...})` starts from the topics you expect. A seed is either a one-sentence
description (`"Sport news: football, rugby, tennis ..."`) or a list of words (`["football", "rugby"]`).

1. **Anchors.** The seed texts are embedded with the model's embedding model. Each document gets the score
   `cos(document, seed)`, for word-list seeds plus `seed_keyword_weight` (0.5) × the share of the seed words
   present in the document. Every document competes for its best seed; the highest-scoring
   `seed_anchors` documents of each seed (default 1% of the corpus, clipped to 5-30) become its anchors.
   A seed that is no document's best match is ignored with a warning, which makes codebook categories that
   are absent from the data visible.
2. **Fixed memberships.** In every Leiden run (consensus, refinement, resolution search) the anchors start
   in one community per seed and are passed to Leiden as `is_membership_fixed`. All other documents move
   freely: they join a seeded community or form new ones.
3. **Naming.** A topic whose anchors mostly come from one seed gets `TopicInfo.seed` and, until an LLM
   relabels it, `label = seed name`. `seed_topics_` and `emergent_topics_` split the result;
   `get_topic_info()` has a `Seed` column; `seed_anchors_` lists the anchor documents. Seeds survive
   `save()`/`load()`.

With pre-computed document embeddings, pass `seed_embeddings` (one vector per seed, same model) or let
TriTopic encode the seeds with `embedding_model`; a dimension mismatch raises an error.

Results: full codebook on the dev splits NMI 0.625 → 0.661, ARI 0.515 → 0.624; held-out ARI 0.399 → 0.498;
81% of the anchors belong to the seed's true class; with half of the classes seeded ARI +17%. BBC demo: NMI
0.742 → 0.801, ARI 0.631 → 0.800 with five one-line seeds. Limitation: with a partial codebook a seeded
topic can absorb an unseeded neighbouring theme (BBC with three seeds: business and politics ended up in one
emergent topic). Seed every theme you know, or run `TopicInterpreter.refine()` afterwards.

## 19. Research toolkit

`tritopic.research` turns a fitted model into the numbers and material a paper needs. Nothing here calls an
LLM unless marked.

| Function | Returns | How |
|---|---|---|
| `topic_reliability(model, method="bootstrap", n_boot=5, sample_frac=0.8)` | `topic, label, size, reliability, core_share, reliable`; sets `model.document_stability_` | Refits on random 80% samples and matches each topic to its best-overlapping refit topic (Jaccard; Greene et al., 2014). `method="consensus"` reuses the Leiden runs: instant but weaker. Topics ≥ 0.7 were 77% pure, topics < 0.5 only 54% (4 corpora). |
| `saturation_curve(model, fractions=None, method="accumulation", min_docs=None)` | `SaturationResult`: `summary`, `point_95`, `point_all`, `novelty`, `late_topics`, `saturated` | Expected share of topics with at least `min_docs` documents in a random subsample, exact via the hypergeometric distribution: a topic accumulation curve. `method="refit"` refits on subsamples instead. Benchmarks: 95% of the topics visible after 25-35% of the data, all after 45-65%. `plot_saturation(result)` |
| `bridge_documents(model, top_n=20, min_share=0.25)` | `doc, topic, other_topic, own_share, bridge_score, text` | Share of a document's nearest-neighbour weight (directed kNN in embedding space) that points into another topic. |
| `topic_connections(model)` | `topic_a, topic_b, strength, bridge_docs` | `W_ab / sqrt(W_a · W_b)` of the cross-topic neighbour weight. |
| `topic_prevalence(model, groups=None, ci=0.95, method="wilson")` | `topic, label, group, n, count, share, ci_low, ci_high` | Wilson score interval, or `method="bootstrap"`. |
| `compare_groups(model, groups, alpha=0.05)` | `share[<group>]` columns, `difference`, `odds_ratio`, `cramers_v`, `p_value`, `p_adjusted`, `significant` | χ² test per topic (Fisher's exact test for 2 groups with small counts), Benjamini-Hochberg across topics. |
| `distinctive_keywords(model, groups, group_a, group_b, topic_id=None, n=10)` | `word, z, typical_for` | Log-odds ratio with informative Dirichlet prior ("Fightin' Words", Monroe et al., 2008), overall or within one topic. |
| `topic_evolution(model, timestamps, freq="Q", link_threshold=0.8)` | `TopicEvolution`: `nodes`, `links`, `events`, `lineage(topic)` | Clusters each period's part of the graph at the model's resolution, links topics of consecutive periods by centroid cosine and labels birth, continuation, split, merge, death. `plot_evolution(evo)` draws a Sankey diagram. |
| `topic_quotes(model, topic_id=None, n=3)` | `topic, rank, quote, doc, score` | Sentences from the topic's most central documents, ranked by similarity to the topic plus a keyword bonus; one quote per document, near-duplicates skipped. Works on lower-cased text. |
| `methods_report(model, corpus="the corpus", fmt="markdown")` | text | Methods paragraph and parameter table written from the fitted model (embedding model, views, resolution choice, quality metrics, software versions). Reproducible, no LLM. |
| `TopicInterpreter.codebook(model, n_quotes=3)` (LLM) | `topic, size, name, definition, inclusion, exclusion, coding_notes, anchor_examples` | One codebook entry per topic for qualitative content analysis (Mayring); exclusion criteria name the neighbouring category; anchors are real quotes. Stored as `model.codebook_`. |
| `intercoder_reliability(model, client)` (LLM) | `IntercoderResult` | See §13. BBC demo: kappa 0.94 on 150 articles. |
