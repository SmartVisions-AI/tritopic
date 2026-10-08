# TriTopic 2.4.0

**Tri-Modal Graph Topic Modeling with Iterative Refinement**

A state-of-the-art topic modeling library that fuses semantic embeddings, lexical similarity, and metadata context through multi-view graph construction, consensus Leiden clustering, and iterative refinement. TriTopic produces stable, interpretable topics and **outperforms BERTopic in head-to-head benchmarks with identical embeddings**.

[![PyPI version](https://badge.fury.io/py/tritopic.svg)](https://badge.fury.io/py/tritopic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/badge/tritopic)](https://pepy.tech/project/tritopic)

> **NMI 0.581 vs. 0.439** (BERTopic, same embeddings) | **2.2x keyword coherence** | **0% outliers** (BERTopic: 18%) | **7x more stable across seeds** | better NMI in **18 of 20** dataset/k settings

---

## Table of Contents

- [Why TriTopic?](#why-tritopic)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Pipeline](#the-pipeline)
- [Configuration Reference](#configuration-reference)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Soft Topic Assignments](#soft-topic-assignments)
- [Cross-Lingual Support](#cross-lingual-support)
- [Hierarchical Topics](#hierarchical-topics)
- [Per-Document Topic Analysis](#per-document-topic-analysis)
- [Outlier Reduction](#outlier-reduction)
- [Topic Merging](#topic-merging)
- [Keyword Extraction](#keyword-extraction)
- [LLM-Powered Labels](#llm-powered-labels)
- [Visualizations](#visualizations)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Comparison with BERTopic](#comparison-with-bertopic)
- [Benchmarks](#benchmarks)
- [Changelog](#changelog)
- [Citation](#citation)
- [License](#license)

---

## Why TriTopic?

Most topic models rely on a single signal -- either word co-occurrences (LDA, NMF) or embeddings alone (BERTopic). This limits their ability to separate topics that share vocabulary but differ semantically, or vice versa.

TriTopic solves this by **fusing three complementary views** of the document corpus into a single graph:

1. **Semantic view** -- sentence-transformer embeddings capture meaning
2. **Lexical view** -- TF-IDF similarity captures surface-level word patterns
3. **Metadata view** -- optional categorical/numerical features add domain context

On top of this multi-view graph, TriTopic applies **consensus Leiden clustering** (multiple runs aggregated via edge-level co-occurrence) and **iterative refinement** (embeddings are pulled toward cluster centroids and re-clustered). The result: topics that are more accurate, more coherent, more stable, and assign every document (zero outliers by default; only clusters smaller than `min_cluster_size` become outliers).

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-view graph fusion** | Combines semantic embeddings, TF-IDF lexical similarity, and optional metadata into a single graph, avoiding the "embedding blur" that single-view models suffer from |
| **Mutual kNN + SNN graphs** | Eliminates noise bridges between unrelated documents using bidirectional neighbor checks and shared-neighbor weighting |
| **Consensus Leiden clustering** | Runs the Leiden algorithm multiple times and merges results via edge-level co-occurrence (Lancichinetti & Fortunato), producing far more stable topics than single-run approaches while scaling linearly with the number of graph edges |
| **Iterative refinement** | Alternates between clustering and embedding refinement, pulling documents toward their topic centroids to sharpen boundaries |
| **Automatic granularity** | `n_topics="auto"` picks the Leiden resolution whose topics have the most coherent keywords, so the topic count adapts to the corpus; `n_topics=k` finds the resolution for exactly *k* topics |
| **Dimensionality reduction** | Reduces high-dimensional embeddings (384-768d) to ~10d with UMAP or PaCMAP before graph construction, improving neighbor quality |
| **100% corpus coverage** | Zero outliers by default -- every document is assigned to a topic, unlike HDBSCAN-based approaches |
| **Soft topic assignments** | Computes per-document probability distributions over all topics, not just hard labels |
| **Cross-lingual support** | Built-in stopwords for English, German, French, and Spanish. `language="multilingual"` auto-selects the `BAAI/bge-m3` embedding model |
| **Hierarchical topics** | Build multi-resolution topic hierarchies with `build_hierarchy()`, split individual topics with `divide()`, and visualize the tree structure |
| **Per-document topic analysis** | Inspect which topics each document belongs to with `get_document_topics()`, compute topic co-occurrence with `topic_overlap_matrix()` |
| **Post-fit outlier reduction** | Reassigns outlier documents using centroid similarity or neighbor voting after the model is fitted |
| **Hierarchical topic merging** | Iteratively merges the most similar topic pairs to reach a target count, or manually merges specific topics |
| **Multiple keyword methods** | Coverage-weighted c-TF-IDF (default), BM25, and KeyBERT keyword extraction |
| **LLM-powered labels** | Generates human-readable topic names via Claude or GPT-4 |
| **Interactive visualizations** | 2D document maps, keyword bar charts, dendrograms, similarity heatmaps, and temporal topic evolution via Plotly |
| **scikit-learn compatible** | Familiar `fit()` / `transform()` / `fit_transform()` API |
| **Save and load** | Full model persistence including fitted reducer, probabilities, and graph state |

---

## Installation

```bash
# Core installation
pip install tritopic

# With LLM labeling support (Claude / GPT-4)
pip install tritopic[llm]

# Full installation (all optional features)
pip install tritopic[full]
```

### From source

```bash
git clone https://github.com/SmartVisions-AI/tritopic.git
cd tritopic
pip install -e ".[dev]"
```

### Dependencies

**Core:** numpy, pandas, scipy, scikit-learn, sentence-transformers, leidenalg, igraph, umap-learn, hdbscan, plotly, tqdm, rank-bm25, keybert

**Optional:** anthropic, openai (for LLM labeling), pacmap, datamapplot (for advanced visualizations)

**Python:** 3.9, 3.10, 3.11, 3.12, 3.13

---

## Quick Start

```python
from tritopic import TriTopic

documents = [
    "Machine learning is transforming healthcare diagnostics",
    "Deep neural networks achieve superhuman performance in image recognition",
    "Climate change affects biodiversity in tropical regions",
    "Renewable energy adoption accelerates globally",
    "The stock market rallied on strong earnings reports",
    # ... hundreds or thousands of documents
]

model = TriTopic(verbose=True)
labels = model.fit_transform(documents)

# View discovered topics
print(model.get_topic_info())
```

**Output:**

```
TriTopic: Fitting model on 1000 documents
   Config: hybrid graph, iterative mode
   -> Generating embeddings (all-MiniLM-L6-v2)...
   -> Reducing dimensions to 10d (umap)...
   -> Building lexical similarity matrix...
   -> Starting iterative refinement (max 5 iterations)...
      Iteration 1...
      Iteration 2...
         ARI vs previous: 0.9234
      Iteration 3...
         ARI vs previous: 0.9812
      Converged at iteration 3
   -> Extracting keywords and representative documents...

Fitting complete!
   Found 12 topics
   47 outlier documents (4.7%)
```

### Post-fit refinement

```python
# Reassign outliers to their nearest topic
model.reduce_outliers(strategy="embeddings")

# Merge down to exactly 8 topics
model.reduce_topics(8)

# Access soft assignments
print(model.probabilities_.shape)   # (n_docs, n_topics)
print(model.probabilities_[0])      # probability distribution for doc 0
```

### Predict new documents

```python
new_docs = [
    "The Mars rover discovered ancient water deposits",
    "Baseball playoffs drew record attendance",
]

# Hard labels
new_labels = model.transform(new_docs)

# Soft probabilities
new_proba = model.transform_proba(new_docs)
```

### Save and load

```python
model.save("my_model.pkl")

from tritopic import TriTopic
loaded = TriTopic.load("my_model.pkl")
```

---

## The Pipeline

TriTopic processes documents through a multi-stage pipeline:

```
Documents
    |
    |--- 1. Embedding Engine ----------------\
    |    (Sentence-BERT / BGE / Instructor)   |
    |                                         |
    |--- 1.5 Dim Reduction (UMAP/PaCMAP) ----+--- Multi-View
    |                                         |    Graph Builder
    |--- 2. Lexical Matrix (TF-IDF) ---------+         |
    |                                         |         |
    \--- 3. Metadata Graph (optional) -------/          |
                                                        v
                                          +-------------------------+
                                          |   Consensus Leiden       |
                                          |   (n runs + co-occur.)   |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Iterative Refinement    |
                                          |  (blend toward centroid) |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Keyword Extraction      |
                                          |  (c-TF-IDF / BM25)      |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Topic Centroids +       |
                                          |  Soft Probabilities      |
                                          +-------------------------+
                                                       |
                                           (optional post-fit)
                                                       |
                                     +---------+-------+--------+
                                     |         |                |
                                reduce_    reduce_        merge_
                                outliers   topics         topics
```

**Step 1 - Embeddings:** Documents are encoded into dense vectors using a sentence-transformer model. You can pass pre-computed embeddings instead.

**Step 1.5 - Dimensionality reduction:** High-dimensional embeddings (384-768d) are projected to ~10 dimensions using UMAP or PaCMAP. This dramatically improves kNN neighbor quality and speeds up graph construction. Full-dimensional embeddings are kept for centroid computation and keyword extraction.

**Step 2 - Lexical matrix:** TF-IDF with n-grams captures surface-level word patterns that embeddings may miss.

**Step 3 - Metadata view (optional):** Categorical and numerical metadata fields are encoded per document. In the fusion step they *reweight* existing semantic/lexical edges (documents with matching metadata get stronger ties); they do not add new edges.

**Step 4 - Multi-view graph fusion:** The semantic kNN graph (built on reduced embeddings with Euclidean distance and a self-tuning Gaussian kernel), the lexical graph, and the metadata view are combined with configurable weights into a single igraph Graph. With `n_topics="auto"`, TriTopic then scans Leiden resolutions on this graph and keeps the coarsest one whose topics have near-maximal keyword coherence. The semantic graph can use mutual kNN, SNN, or a hybrid of both.

**Step 5 - Consensus Leiden clustering:** The Leiden algorithm runs multiple times (default: 10) with different seeds. For every graph edge, TriTopic records how often its two documents land in the same cluster. Edges with low agreement are dropped, the rest are reweighted by agreement, and Leiden is re-run on this consensus graph until the runs agree. This produces a partition that is more stable than any single run, in O(edges) instead of O(n²) memory. Clusters below `min_cluster_size` are marked as outliers (-1).

**Step 6 - Iterative refinement:** The embeddings used for graph building (the reduced embeddings if dimensionality reduction is on) are softly blended toward their topic centroid (30% pull, decaying to 10%; core members are pulled more than borderline ones), then the graph and clustering are re-run. This loop continues until the Adjusted Rand Index between consecutive iterations exceeds the convergence threshold (default: 0.95), or until `max_iterations` is reached. The lexical graph is built once and reused.

**Step 7 - Keywords and centroids:** Coverage-weighted c-TF-IDF (or BM25/KeyBERT) extracts representative keywords per topic; the corpus is tokenized once and the counts are shared with the lexical view. Topic centroids are computed as the mean embedding of each topic's documents. Soft probabilities are computed via cosine similarity to centroids passed through softmax.

---

## Configuration Reference

All parameters are set through `TriTopicConfig` or as constructor overrides:

```python
from tritopic import TriTopic, TriTopicConfig

config = TriTopicConfig(
    # --- Language ---
    language="english",                    # "english", "german", "french", "spanish", "multilingual"

    # --- Embedding ---
    embedding_model="all-MiniLM-L6-v2",   # sentence-transformers model name
    embedding_batch_size=32,               # encoding batch size

    # --- Dimensionality Reduction ---
    use_dim_reduction=True,                # reduce before graph building
    reduced_dims=10,                       # target dimensionality
    dim_reduction_method="umap",           # "umap" or "pacmap"
    umap_n_neighbors=15,                   # UMAP/PaCMAP neighbor count
    umap_min_dist=0.0,                     # 0.0 optimized for clustering
    reduced_metric="euclidean",            # kNN metric on reduced embeddings

    # --- Graph Construction ---
    n_neighbors=15,                        # k for kNN graph
    metric="cosine",                       # kNN metric on full embeddings (no dim reduction)
    graph_type="hybrid",                   # "knn", "mutual_knn", "snn", "hybrid"
    snn_weight=0.5,                        # SNN weight in hybrid mode

    # --- Multi-View Fusion ---
    use_lexical_view=True,                 # include TF-IDF view
    use_metadata_view=False,               # include metadata view
    semantic_weight=0.5,                   # weight for semantic graph
    lexical_weight=0.3,                    # weight for lexical graph
    metadata_weight=0.2,                   # weight for metadata graph

    # --- Clustering ---
    resolution=0.3,                        # Leiden resolution (start for n_topics=k; used for "auto" if auto_resolution=False)
    auto_resolution=True,                  # n_topics="auto": choose resolution by keyword coherence
    resolution_range=None,                 # search range for auto_resolution (default (0.01, 1.0))
    auto_resolution_steps=15,              # resolutions scanned
    auto_resolution_tolerance=0.05,        # pick the coarsest within 5% of the best coherence
    auto_resolution_max_share=0.5,         # ignore partitions where one topic holds >50% of docs
    n_consensus_runs=10,                   # number of Leiden runs for consensus
    min_cluster_size=5,                    # clusters smaller than this become outliers

    # --- Iterative Refinement ---
    use_iterative_refinement=True,         # enable the refinement loop
    max_iterations=5,                      # maximum refinement iterations
    convergence_threshold=0.95,            # ARI threshold to stop early

    # --- Keyword Extraction ---
    n_keywords=10,                         # keywords per topic
    n_representative_docs=5,               # representative docs per topic
    keyword_method="ctfidf",               # "ctfidf", "bm25", or "keybert"

    # --- Soft Assignment ---
    soft_assignment_method="centroid",     # "centroid" or "graph"

    # --- Outlier Handling ---
    outlier_threshold=0.35,                # cosine similarity threshold for transform()
    softmax_temperature=5.0,               # higher = sharper topic probabilities

    # --- Misc ---
    random_state=42,
    verbose=True,
)

model = TriTopic(config=config)
```

### Topic granularity

- `n_topics="auto"` (default): TriTopic scans `auto_resolution_steps` Leiden resolutions, extracts keywords for each candidate partition, and keeps the coarsest resolution whose mean keyword coherence (NPMI) is within `auto_resolution_tolerance` of the best. Partitions in which one topic holds more than `auto_resolution_max_share` of the documents are skipped (very coarse partitions score high NPMI on generic words). The scan is available as `model.resolution_search_` (resolution, n_topics, coherence), the chosen value as `model.resolution_`.
- `n_topics=k`: bisection over the resolution until exactly *k* topics remain (merging down if needed).
- Fixed resolution: `TriTopicConfig(auto_resolution=False, resolution=...)`.

**Quick overrides** without creating a config object:

```python
model = TriTopic(
    embedding_model="all-mpnet-base-v2",
    n_neighbors=20,
    use_iterative_refinement=True,
    verbose=True,
    random_state=42,
)
```

You can also modify the config after construction:

```python
model = TriTopic()
model.config.use_dim_reduction = False       # disable dim reduction
model.config.graph_type = "snn"              # use pure SNN graph
model.config.keyword_method = "bm25"         # switch keyword method
```

---

## Dimensionality Reduction

kNN graphs built on high-dimensional embeddings (384-768d) suffer from the curse of dimensionality: distances concentrate and neighbor quality degrades. TriTopic addresses this by reducing embeddings to a low-dimensional space before graph construction.

```python
model = TriTopic()
model.config.use_dim_reduction = True        # enabled by default
model.config.reduced_dims = 10               # target dimensions
model.config.dim_reduction_method = "umap"   # or "pacmap"
model.config.umap_n_neighbors = 15
model.config.umap_min_dist = 0.0             # 0.0 is best for clustering

model.fit(documents)

# Reduced embeddings are stored alongside full embeddings
print(model.reduced_embeddings_.shape)  # (n_docs, 10)
print(model.embeddings_.shape)          # (n_docs, 384)  full embeddings kept (unrefined)
```

**How it works:**

- Reduced embeddings are used only for graph construction (kNN neighbor search, Euclidean distance -- UMAP/PaCMAP output is a Euclidean layout, so cosine angles around its arbitrary origin are not meaningful)
- Full-dimensional embeddings are used for centroid computation, keyword extraction, representative docs, and similarity calculations
- Iterative refinement operates directly on the reduced embeddings (no re-projection through the reducer)
- The fitted reducer is saved with `model.save()` so `transform()` on new documents works correctly

**When to disable it:**

```python
model.config.use_dim_reduction = False
```

Disable if your embeddings are already low-dimensional, or if you want to experiment with raw high-dimensional graph construction.

---

## Soft Topic Assignments

Every document gets a probability distribution over all topics, not just a hard label.

### Training documents

After `fit()`, probabilities are automatically available:

```python
model.fit(documents)

# Shape: (n_documents, n_topics)
print(model.probabilities_.shape)

# Each row sums to ~1.0
print(model.probabilities_[0].sum())  # ~1.0

# Probability distribution for document 0
for i, prob in enumerate(model.probabilities_[0]):
    topic_id = [t.topic_id for t in model.topics_ if t.topic_id != -1][i]
    print(f"  Topic {topic_id}: {prob:.3f}")
```

### New documents

```python
proba = model.transform_proba(["A new document about space exploration"])
# Shape: (1, n_topics)
print(proba)
```

**How it works:** Cosine similarity between document embeddings and topic centroid embeddings, followed by softmax normalization. Probabilities are recomputed automatically after any post-fit operation (outlier reduction, topic merging).

You can also use graph-based soft assignments, which derive probabilities from a document's graph neighborhood:

```python
model = TriTopic(soft_assignment_method="graph")
model.fit(documents)
```

---

## Cross-Lingual Support

TriTopic supports topic modeling in multiple languages. The `language` parameter controls stopword filtering and, when set to `"multilingual"`, automatically selects an appropriate embedding model.

### Single language

```python
# German corpus
model = TriTopic(language="german")
model.fit_transform(german_documents)

# French corpus
model = TriTopic(language="french")
model.fit_transform(french_documents)
```

Built-in stopwords are provided for English, German, French, and Spanish. Stopwords are used during keyword extraction to filter out common function words.

### Multilingual corpus

```python
# Mixed-language corpus — auto-selects BAAI/bge-m3 embeddings
model = TriTopic(language="multilingual")
model.fit_transform(mixed_language_documents)
```

When `language="multilingual"` and the default embedding model is used, TriTopic automatically switches to `BAAI/bge-m3` (1024 dimensions), which supports 100+ languages. Stopword filtering is disabled in multilingual mode to avoid language-specific bias.

You can still use a custom embedding model with multilingual mode:

```python
model = TriTopic(language="multilingual", embedding_model="your/custom-model")
```

### Supported languages

| Language | Stopwords | Auto-model |
|---|---|---|
| `"english"` (default) | sklearn built-in | `all-MiniLM-L6-v2` |
| `"german"` | Built-in (250+ words) | `all-MiniLM-L6-v2` |
| `"french"` | Built-in | `all-MiniLM-L6-v2` |
| `"spanish"` | Built-in | `all-MiniLM-L6-v2` |
| `"multilingual"` | Disabled | `BAAI/bge-m3` (auto) |

---

## Hierarchical Topics

TriTopic can build a multi-resolution topic hierarchy, allowing you to explore topics at different levels of granularity.

### Build a hierarchy

```python
model.fit(documents)

# Build a 3-level hierarchy (coarse → medium → fine)
hierarchy = model.build_hierarchy(n_levels=3)
print(hierarchy)  # TopicHierarchy(levels=3, topics_per_level=[3, 8, 15])
```

The hierarchy is constructed by clustering at multiple Leiden resolutions (automatically spaced from coarse to fine). Each fine-grained topic is linked to its coarse-grained parent via majority-vote assignment.

### Navigate the hierarchy

```python
# Access topics at a specific level
coarse_topics = hierarchy.cut(0)   # broad themes
fine_topics = hierarchy.cut(2)     # specific sub-topics

# Traverse the tree
for root in hierarchy.roots:
    print(f"{root.node_id}: {root.keywords[:3]}")
    for child in root.children:
        print(f"  {child.node_id}: {child.keywords[:3]}")

# Look up a specific node
node = hierarchy.get_node("L0_3")
```

### Explicit resolution levels

```python
# Provide your own resolution values (coarse → fine)
hierarchy = model.build_hierarchy(resolution_levels=[0.25, 1.0, 4.0])
```

### Divide a single topic

Split one topic into finer sub-topics without rebuilding the full hierarchy:

```python
subtopics = model.divide(topic_id=0, n_subtopics=3)
for st in subtopics:
    print(f"  Sub-topic {st.topic_id}: {st.keywords[:5]}")
```

This extracts the subgraph for the given topic, runs Leiden at higher resolution, and updates `model.labels_` in-place. Keywords, centroids, and probabilities are refreshed automatically.

### Visualize the hierarchy

```python
model.build_hierarchy(n_levels=3)
fig = model.visualize_hierarchy_tree()
fig.show()
```

---

## Per-Document Topic Analysis

Inspect which topics each document belongs to and how topics overlap across the corpus.

### Top topics for a document

```python
# Get the top 3 topics for document 0
topics = model.get_document_topics(doc_idx=0, top_n=3)
for topic_id, probability in topics:
    print(f"  Topic {topic_id}: {probability:.4f}")

# Use graph-based method explicitly
topics = model.get_document_topics(doc_idx=0, top_n=3, method="graph")
```

### Topic overlap matrix

Compute how often topics co-occur across documents. A topic is considered "active" for a document when its probability exceeds the threshold:

```python
overlap = model.topic_overlap_matrix(threshold=0.1)
print(overlap)  # Symmetric DataFrame (n_topics x n_topics)
```

The diagonal shows how many documents strongly belong to each topic. Off-diagonal values reveal topic pairs that frequently co-occur.

### Visualize overlap

```python
fig = model.visualize_overlap(threshold=0.1)
fig.show()
```

---

## Outlier Reduction

Clusters smaller than `min_cluster_size` are marked as outliers (-1); with the default settings this is rare. `reduce_outliers()` reassigns them post-fit.

### Strategy: embeddings (default)

Each outlier is assigned to the topic whose centroid is most similar, if the similarity exceeds a threshold:

```python
model.fit(documents)
print(f"Outliers before: {(model.labels_ == -1).sum()}")

# Default threshold = config.outlier_threshold (0.35)
model.reduce_outliers(strategy="embeddings")
print(f"Outliers after: {(model.labels_ == -1).sum()}")

# Lower threshold = more aggressive reassignment
model.reduce_outliers(strategy="embeddings", threshold=0.05)
```

### Strategy: neighbors

Each outlier is assigned by majority vote of its k nearest non-outlier neighbors:

```python
model.reduce_outliers(strategy="neighbors")
```

This strategy is threshold-free and works well when outliers are near cluster boundaries.

**After reassignment:** Keywords, centroids, topic sizes, and probabilities are all recomputed automatically.

---

## Topic Merging

### Automatic: reduce to a target count

`reduce_topics()` iteratively merges the two most cosine-similar topic centroids until the target count is reached:

```python
model.fit(documents)
print(f"Topics found: {len([t for t in model.topics_ if t.topic_id != -1])}")

# Reduce to exactly 5 topics
model.reduce_topics(5)
print(f"Topics after: {len([t for t in model.topics_ if t.topic_id != -1])}")
```

At each step, the two most similar centroids are found (similarity is mildly penalized for very unequal topic sizes), and the smaller topic is relabeled to the larger one. After all merges complete, keywords and centroids are re-extracted.

### Manual: merge specific topics

```python
# Merge topics 2 and 7 into one (the larger one's ID is kept)
model.merge_topics([2, 7])

# Merge three topics together
model.merge_topics([1, 4, 9])
```

This is useful when you inspect topics and find two that clearly cover the same theme.

---

## Keyword Extraction

TriTopic supports three keyword extraction methods:

### c-TF-IDF (default)

Class-based TF-IDF treats all documents in a topic as a single "class document". TriTopic weights the term frequency by *document coverage*:

`score(t) = IDF(t) * sqrt(tf_share(t) * coverage(t))`

where `tf_share` is the share of the topic's tokens that are *t* and `coverage` the fraction of the topic's documents containing *t*. Term frequency alone favours words repeated in a few long documents; coverage alone favours boilerplate. The geometric mean keeps terms that are both frequent and spread across the topic (+45% NPMI coherence over plain c-TF-IDF on the dev benchmarks). Tokens must start with two letters, so numbers and underscores (`000`, `__`) never become keywords.

```python
model.config.keyword_method = "ctfidf"
```

### BM25

Ranks terms by their average BM25 weight within the topic's documents relative to the corpus average, times log(1 + frequency). More robust to document length variations than TF-IDF:

```python
model.config.keyword_method = "bm25"
```

### KeyBERT

Embedding-based extraction that finds keywords by comparing candidate n-gram embeddings to the topic embedding. Uses Maximal Marginal Relevance (MMR) for diversity:

```python
model.config.keyword_method = "keybert"
```

### Accessing keywords

```python
# DataFrame view
df = model.get_topic_info()
print(df[["Topic", "Size", "Keywords"]])

# Detailed access for a specific topic
topic = model.get_topic(0)
print(topic.keywords)         # ['machine', 'learning', 'neural', ...]
print(topic.keyword_scores)   # [0.42, 0.38, 0.31, ...]

# Representative documents
docs = model.get_representative_docs(0, n_docs=3)
for idx, text in docs:
    print(f"  Doc {idx}: {text[:100]}...")
```

---

## LLM-Powered Labels

Generate human-readable topic names using Claude or GPT-4.

### With Claude (Anthropic)

```python
from tritopic import TriTopic, LLMLabeler

model = TriTopic()
model.fit(documents)

labeler = LLMLabeler(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-haiku-4-5",          # fast and cheap (default)
    language="english",                 # output language
    domain_hint="technology news",      # optional domain context
)
model.generate_labels(labeler)

# Topics now have labels and descriptions
df = model.get_topic_info()
print(df[["Topic", "Label", "Description"]])
```

### With GPT-4 (OpenAI)

```python
labeler = LLMLabeler(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    language="german",          # works in any language
)
model.generate_labels(labeler)
```

### Simple labeler (no API needed)

```python
from tritopic import SimpleLabeler

labeler = SimpleLabeler(n_words=3)
model.generate_labels(labeler)
# Labels like "Machine & Learning & Neural"
```

### Label specific topics only

```python
model.generate_labels(labeler, topics=[0, 3, 5])
```

If the LLM API call fails, the labeler falls back to a keyword-based label automatically.

---

## Visualizations

All visualizations return interactive Plotly figures.

### Document map

2D scatter plot where each point is a document, colored by topic:

```python
fig = model.visualize(method="umap", show_outliers=True)
fig.show()
fig.write_html("document_map.html")
```

### Topic keywords

Horizontal bar charts showing the top keywords and their scores for each topic:

```python
fig = model.visualize_topics(n_keywords=8)
fig.show()
```

### Topic hierarchy

Dendrogram showing how topics relate to each other based on centroid distances:

```python
fig = model.visualize_hierarchy()
fig.show()
```

### Topic similarity heatmap

Cosine similarity matrix between all topic centroids:

```python
from tritopic import TopicVisualizer

viz = TopicVisualizer()
fig = viz.plot_topic_similarity(model.topic_embeddings_, model.topics_)
fig.show()
```

### Topics over time

Stacked area chart showing topic prevalence over time (requires timestamps):

```python
from tritopic import TopicVisualizer

viz = TopicVisualizer()
fig = viz.plot_topic_over_time(
    labels=model.labels_,
    timestamps=your_timestamps,   # list of datetime-like values
    topics=model.topics_,
)
fig.show()
```

---

## Evaluation

```python
metrics = model.evaluate()
```

Returns a dictionary with:

| Metric | Range | Description |
|---|---|---|
| `coherence_mean` | -1 to 1 | Average NPMI coherence across topics, whole corpus as reference (higher = more coherent keywords) |
| `coherence_std` | 0+ | Standard deviation of coherence across topics |
| `diversity` | 0 to 1 | Proportion of unique keywords across all topics (higher = more distinct topics) |
| `stability` | -1 to 1 | Average pairwise ARI across consensus runs (higher = more reproducible) |
| `n_topics` | 1+ | Number of non-outlier topics |
| `outlier_ratio` | 0 to 1 | Fraction of documents labeled as outliers |

Coherence is computed on document-level co-occurrence over the **whole corpus**, with the same analyzer as keyword extraction (bigram keywords are counted; pairs that never co-occur score -1). Values are not comparable with versions before 2.4.0, which used only each topic's own documents as reference and therefore reported inflated scores.

Additional metrics are available as standalone functions:

```python
from tritopic.utils.metrics import (
    compute_coherence,
    compute_coherence_batch,   # many topics, one pass over the corpus
    compute_diversity,
    compute_stability,
    compute_silhouette,
    compute_downstream_score,
)

# Silhouette score for cluster separation
sil = compute_silhouette(model.embeddings_, model.labels_)

# Downstream classification performance
f1 = compute_downstream_score(
    model.embeddings_, model.labels_, true_labels, task="classification"
)
```

---

## Advanced Usage

### Pre-computed embeddings

Skip the embedding step by passing your own vectors:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = encoder.encode(documents)

model = TriTopic()
model.fit(documents, embeddings=embeddings)
```

### Multi-model embeddings

Combine embeddings from multiple models for richer representations:

```python
from tritopic import EmbeddingEngine
from tritopic.core.embeddings import MultiModelEmbedding

multi = MultiModelEmbedding(
    model_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    weights=[0.5, 0.5],
)
embeddings = multi.encode(documents)

model = TriTopic()
model.fit(documents, embeddings=embeddings)
```

### Metadata-enhanced topics

Documents with shared metadata (source, category, date) get stronger ties in the graph:

```python
import pandas as pd

metadata = pd.DataFrame({
    "source": ["twitter", "news", "twitter", ...],
    "category": ["tech", "science", "tech", ...],
})

model = TriTopic()
model.config.use_metadata_view = True
model.config.metadata_weight = 0.2

model.fit(documents, metadata=metadata)
```

Metadata reweights edges that already exist in the semantic/lexical graph; it never adds new ones. Per edge, the metadata similarity is the average over columns of: exact match for categorical columns (strings, categories, booleans), and `1 - |difference|` for numerical/datetime columns after min-max normalization (counted only if > 0.8). Connecting *every* pair of documents with the same category would create O(n²) edges and let the metadata dominate the topics.

### Target number of topics

Use `n_topics` to automatically find the Leiden resolution that produces a specific number of topics:

```python
model = TriTopic(n_topics=10)
model.fit(documents)
# Bisection over the resolution (log-space, counting only clusters >=
# min_cluster_size); merges down if it overshoots
```

### Finding the optimal resolution

The resolution parameter controls how many topics Leiden produces. You can search for the best value:

```python
from tritopic.core.clustering import ConsensusLeiden

# After initial fit
clusterer = ConsensusLeiden()
optimal = clusterer.find_optimal_resolution(
    graph=model.graph_,
    resolution_range=(0.05, 2.0),
    n_steps=10,
    target_n_topics=15,       # optional: aim for ~15 topics
    min_cluster_size=5,       # only count clusters that become topics
)
print(f"Optimal resolution: {optimal}")

# Re-fit with the optimal resolution
model.config.resolution = optimal
model.fit(documents)
```

### Disabling features

```python
# No iterative refinement (faster, less accurate)
model = TriTopic(use_iterative_refinement=False)

# No dimensionality reduction
model.config.use_dim_reduction = False

# No lexical view (embeddings only)
model.config.use_lexical_view = False
```

### Complete workflow

```python
from tritopic import TriTopic, TriTopicConfig, LLMLabeler

# 1. Configure
config = TriTopicConfig(
    embedding_model="all-mpnet-base-v2",
    n_neighbors=20,
    graph_type="hybrid",
    use_dim_reduction=True,
    reduced_dims=10,
    n_consensus_runs=15,
    use_iterative_refinement=True,
    max_iterations=7,
    convergence_threshold=0.97,
    keyword_method="ctfidf",
    n_keywords=15,
)

# 2. Fit
model = TriTopic(config=config)
model.fit(documents)

# 3. Reduce outliers
model.reduce_outliers(strategy="embeddings", threshold=0.05)

# 4. Merge to desired granularity
model.reduce_topics(10)

# 5. Label with LLM
labeler = LLMLabeler(provider="anthropic", api_key="...")
model.generate_labels(labeler)

# 6. Evaluate
metrics = model.evaluate()

# 7. Explore
print(model.get_topic_info())
print(f"Probabilities shape: {model.probabilities_.shape}")

fig = model.visualize()
fig.show()

# 8. Save
model.save("production_model.pkl")
```

---

## API Reference

### TriTopic

The main model class. Follows the scikit-learn fit/transform pattern.

| Method | Description |
|---|---|
| `fit(documents, embeddings?, metadata?)` | Fit the model. Returns `self`. |
| `fit_transform(documents, embeddings?, metadata?)` | Fit and return hard labels. |
| `transform(documents)` | Assign topics to new documents. Returns labels array. |
| `transform_proba(documents)` | Get soft probabilities for new documents. Returns `(n_docs, n_topics)` matrix. |
| `reduce_outliers(strategy?, threshold?)` | Reassign outliers. Strategies: `"embeddings"`, `"neighbors"`. Returns `self`. |
| `reduce_topics(n_topics)` | Merge down to `n_topics` non-outlier topics. Returns `self`. |
| `merge_topics(topics_to_merge)` | Merge specific topic IDs into one. Returns `self`. |
| `get_topic_info()` | DataFrame with Topic, Size, Keywords, Label, Coherence columns. |
| `get_topic(topic_id)` | Get `TopicInfo` for a specific topic. |
| `get_representative_docs(topic_id, n_docs?)` | Get `(index, text)` tuples for a topic's most central documents. |
| `generate_labels(labeler, topics?)` | Generate LLM labels for topics. |
| `evaluate()` | Compute coherence, diversity, stability, and outlier ratio. |
| `build_hierarchy(resolution_levels?, n_levels?)` | Build multi-resolution topic hierarchy. Returns `TopicHierarchy`. |
| `divide(topic_id, n_subtopics?)` | Split a single topic into finer sub-topics. Returns `list[TopicInfo]`. |
| `get_document_topics(doc_idx, top_n?, method?)` | Get top-N topics for a document with probabilities. |
| `topic_overlap_matrix(threshold?)` | Topic co-occurrence matrix as DataFrame. |
| `visualize(method?, show_outliers?, ...)` | 2D document scatter plot. |
| `visualize_topics(n_keywords?, ...)` | Keyword bar charts per topic. |
| `visualize_hierarchy(...)` | Topic dendrogram (centroid distances). |
| `visualize_hierarchy_tree(...)` | Multi-level hierarchy tree (requires `build_hierarchy()`). |
| `visualize_overlap(threshold?, ...)` | Topic overlap heatmap. |
| `save(path)` | Pickle model to disk (includes all state, reducer, probabilities). |
| `TriTopic.load(path)` | Class method to load a saved model. |

### Key attributes after fit

| Attribute | Type | Description |
|---|---|---|
| `labels_` | `np.ndarray` | Hard topic assignment per document. -1 = outlier. |
| `probabilities_` | `np.ndarray` | Soft assignments, shape `(n_docs, n_topics)`. Rows sum to ~1. |
| `embeddings_` | `np.ndarray` | Full-dimensional document embeddings (refined only if iterative refinement runs without dim reduction). |
| `original_embeddings_` | `np.ndarray` | Unrefined embeddings; used for centroids, probabilities, outlier reduction, and merging. |
| `reduced_embeddings_` | `np.ndarray` | Low-dimensional embeddings used for graph building (refined if iterative). |
| `resolution_` | `float` | Leiden resolution used by `fit()` (chosen automatically with `n_topics="auto"`). |
| `resolution_search_` | `list[tuple]` | Auto-resolution scan: (resolution, n_topics, coherence). |
| `topic_embeddings_` | `np.ndarray` | Centroid embedding per topic, shape `(n_topics, embed_dim)`. |
| `topics_` | `list[TopicInfo]` | List of `TopicInfo` objects with keywords, scores, centroids. |
| `documents_` | `list[str]` | Stored training documents. |
| `graph_` | `igraph.Graph` | The final fused graph. |
| `hierarchy_` | `TopicHierarchy \| None` | Multi-resolution hierarchy (set by `build_hierarchy()`). |

### TopicInfo

| Field | Type | Description |
|---|---|---|
| `topic_id` | `int` | Topic ID (-1 for outliers). |
| `size` | `int` | Number of documents in the topic. |
| `keywords` | `list[str]` | Ranked keywords. |
| `keyword_scores` | `list[float]` | Keyword importance scores. |
| `representative_docs` | `list[int]` | Indices of documents closest to centroid. |
| `label` | `str \| None` | LLM-generated label. |
| `description` | `str \| None` | LLM-generated description. |
| `centroid` | `np.ndarray \| None` | Topic centroid embedding. |
| `coherence` | `float \| None` | NPMI coherence score (after `evaluate()`). |

### Supporting classes

| Class | Module | Purpose |
|---|---|---|
| `TriTopicConfig` | `tritopic.core.model` | All configuration parameters (see [Configuration Reference](#configuration-reference)) |
| `EmbeddingEngine` | `tritopic.core.embeddings` | Encode documents with sentence-transformers. Supports Instructor and BGE models. |
| `MultiModelEmbedding` | `tritopic.core.embeddings` | Combine embeddings from multiple models. |
| `GraphBuilder` | `tritopic.core.graph_builder` | Build kNN, mutual kNN, SNN, hybrid, lexical, and metadata graphs. |
| `ConsensusLeiden` | `tritopic.core.clustering` | Leiden clustering with consensus and resolution search. |
| `HDBSCANClusterer` | `tritopic.core.clustering` | Alternative HDBSCAN clustering. |
| `TopicNode` | `tritopic.core.hierarchy` | A node in the topic hierarchy with keywords, children, and document indices. |
| `TopicHierarchy` | `tritopic.core.hierarchy` | Multi-resolution topic tree with `cut()`, `flatten()`, `get_node()` methods. |
| `KeywordExtractor` | `tritopic.core.keywords` | c-TF-IDF, BM25, and KeyBERT keyword extraction. |
| `KeyphraseExtractor` | `tritopic.core.keywords` | Multi-word keyphrase extraction (YAKE). |
| `LLMLabeler` | `tritopic.labeling.llm_labeler` | Generate labels via Claude or GPT-4. |
| `SimpleLabeler` | `tritopic.labeling.llm_labeler` | Rule-based labels from top keywords. |
| `TopicVisualizer` | `tritopic.visualization.plotter` | All Plotly visualizations. |

---

## Architecture

### Graph types

**kNN:** Each document connects to its k nearest neighbors. Simple but includes asymmetric "one-way" connections that can bridge unrelated clusters.

**Mutual kNN:** Only keeps edges where both nodes are in each other's neighborhoods. This removes noise bridges and produces cleaner clusters.

**SNN (Shared Nearest Neighbors):** Edge weight equals the number of shared neighbors between two nodes, normalized by k. This captures structural similarity and is robust against noise.

**Hybrid (default):** Weighted combination of mutual kNN and SNN: `(1 - snn_weight) * mutual_kNN + snn_weight * SNN`. Gives both direct similarity (mutual kNN) and structural similarity (SNN).

### Consensus clustering

Running Leiden once is sensitive to random initialization. TriTopic runs it `n_consensus_runs` times (default: 10) with different seeds. For every graph edge it records the fraction of runs in which both endpoints share a cluster (Lancichinetti & Fortunato, 2012). Edges below 50% agreement are dropped (each node keeps its most consistent edge), the rest are reweighted by agreement, and Leiden is re-run on this consensus graph until all runs agree or the consensus graph stops changing; the run with the highest average ARI to the others is returned. This needs only O(edges) memory. The stability score (average pairwise ARI across the initial runs) quantifies how reproducible the clustering is.

### Iterative refinement

After an initial clustering pass, the graph-building embeddings are softly blended toward their topic centroid: `refined = (1 - b) * x + b * centroid`, where `b` decays from 0.3 to 0.1 over the iterations and is scaled down for documents far from the centroid. The semantic graph and clustering re-run on the refined embeddings. This process converges when consecutive partitions have ARI >= 0.95 (configurable). The effect is tighter, more separated topic clusters.

### Supported embedding models

Any model from the [sentence-transformers](https://www.sbert.net/) library works. Recommended choices:

| Model | Dimensions | Speed | Quality | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default. Best speed/quality tradeoff. |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality, 2x slower. |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Best | State-of-the-art for English. |
| `BAAI/bge-m3` | 1024 | Slow | Best | Multilingual support. Auto-selected with `language="multilingual"`. |
| `hkunlp/instructor-large` | 768 | Slow | Best | Task-specific with instructions. |

---

## Comparison with BERTopic

| Aspect | BERTopic | TriTopic |
|---|---|---|
| **Graph construction** | kNN only | Mutual kNN + SNN hybrid |
| **Dimensionality reduction** | UMAP (for clustering) | UMAP/PaCMAP (configurable) |
| **Clustering** | HDBSCAN (single run) | Leiden with consensus (n runs) |
| **Stability** | Low (varies between runs) | High (consensus + stability score) |
| **Input signals** | Embeddings only | Semantic + Lexical + Metadata |
| **Refinement** | None | Iterative embedding refinement |
| **Coverage** | ~82% (18.4% outliers avg.) | **100%** (0% outliers) |
| **Soft assignments** | Via HDBSCAN probabilities | Cosine similarity + softmax |
| **Outlier reduction** | 4 strategies | 2 strategies (embeddings, neighbors) |
| **Topic merging** | Hierarchical | Hierarchical + manual merge |
| **Keyword extraction** | c-TF-IDF | c-TF-IDF, BM25, or KeyBERT |
| **LLM labels** | Via representation model | Built-in Claude/GPT-4 support |
| **Cross-lingual** | Manual model selection | Built-in `language` param with auto-model selection |
| **Hierarchical topics** | Hierarchical topic modeling | Multi-resolution hierarchy + single-topic `divide()` |
| **NMI (benchmark avg., fixed k)** | 0.439 | **0.581 (+32%)** |
| **Coherence (strict NPMI, fixed k)** | 0.117 (tuned) | **0.263 (2.2x)** |
| **Seed stability (NMI spread)** | 0.105 | **0.014** |

---

## Benchmarks

Head-to-head comparison with BERTopic 0.17.4 on the four datasets of the original TriTopic benchmark
(20 Newsgroups 2,000 docs, BBC News 1,225, AG News 2,000, arXiv 2,000), 5 topic counts per dataset plus
automatic mode, 3 seeds each.

- **Same embeddings for every model** (`all-MiniLM-L6-v2`, pre-computed once per dataset).
- **BERTopic default** and **BERTopic tuned** (its documented keyword best practice:
  `CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)` +
  `ClassTfidfTransformer(reduce_frequent_words=True)`); UMAP seed fixed for reproducibility.
- **No tuning on the evaluation data**: all 2.4 design decisions were made on disjoint development splits
  (20NG test, BBC test, AG News train sample, arXiv validation); the evaluation splits were run once.
- **Metrics**: NMI/ARI on all documents (outliers form one class); strict NPMI over the whole corpus (pairs that
  never co-occur score -1); "2.3 benchmark" NPMI = the lenient coherence of the previous benchmark; fit time with
  pre-computed embeddings, runs executed sequentially on one machine.

#### Fixed topic count (k from the paper grid)

| Model | NMI | ARI | NPMI (strict) | NPMI (2.3 benchmark) | Diversity | Outliers | Fit time (s) |
|---|---|---|---|---|---|---|---|
| **TriTopic 2.4** | 0.581 | 0.438 | 0.263 | 0.355 | 0.94 | 0.0% | 9.8 |
| TriTopic 2.3 | 0.581 | 0.436 | 0.212 | 0.343 | 0.92 | 0.0% | 44.3 |
| BERTopic (default) | 0.439 | 0.285 | 0.048 | 0.212 | 0.44 | 18.4% | 9.9 |
| BERTopic (tuned) | 0.439 | 0.285 | 0.117 | 0.294 | 0.89 | 18.4% | 16.6 |

#### Automatic topic count (`n_topics="auto"` / BERTopic default)

| Model | NMI | ARI | NPMI (strict) | NPMI (2.3 benchmark) | Diversity | Outliers | Fit time (s) | Topics found |
|---|---|---|---|---|---|---|---|---|
| **TriTopic 2.4** | 0.560 | 0.398 | 0.299 | 0.397 | 0.94 | 0.0% | 11.4 | 21.4 |
| TriTopic 2.3 | 0.528 | 0.302 | 0.200 | 0.391 | 0.90 | 0.0% | 36.7 | 30.0 |
| BERTopic (default) | 0.443 | 0.247 | 0.109 | 0.263 | 0.54 | 18.4% | 8.3 | 26.8 |
| BERTopic (tuned) | 0.443 | 0.247 | 0.123 | 0.387 | 0.89 | 18.4% | 11.7 | 26.8 |

#### NMI per dataset (fixed k)

| Dataset | **TriTopic 2.4** | TriTopic 2.3 | BERTopic (default) | BERTopic (tuned) |
|---|---|---|---|---|
| 20 Newsgroups | 0.529 | **0.530** | 0.264 | 0.264 |
| AG News | 0.525 | **0.527** | 0.377 | 0.377 |
| arXiv | **0.566** | 0.565 | 0.450 | 0.450 |
| BBC News | **0.702** | 0.701 | 0.663 | 0.663 |

#### Seed stability (fixed k): NMI spread across 3 seeds

| Model | Mean spread | Worst spread |
|---|---|---|
| **TriTopic 2.4** | 0.014 | 0.046 |
| TriTopic 2.3 | 0.014 | 0.048 |
| BERTopic (default) | 0.105 | 0.318 |
| BERTopic (tuned) | 0.105 | 0.318 |

Against BERTopic (default and tuned), TriTopic 2.4 has the better NMI in **18 of 20** dataset/k combinations.

**Where TriTopic does not lead:** on BBC News, BERTopic has the higher ARI for k = 5-20 (NMI is close), and on
short texts BERTopic is about 1-2 s per run faster. In automatic mode TriTopic tends to find more topics than
there are classes on AG News and BBC; pass `n_topics` if you know the granularity you need.

Reproduce with `python benchmarks/compare_bertopic.py --pkg . --models tritopic,bertopic,bertopic_tuned --out results.csv`,
then `python benchmarks/summarize.py results.csv` and `python benchmarks/results_to_markdown.py tables.md results.csv`.

> Earlier releases quoted mean NMI 0.575 vs. 0.513 for BERTopic. That benchmark gave BERTopic
> `all-mpnet-base-v2` embeddings while TriTopic used MiniLM, and its coherence skipped word pairs that never
> co-occur; those figures are superseded by the table above.

---

## Changelog

### 2.4.0

**Performance** (fit time 4.5x faster than 2.3 at equal NMI; arXiv 102 s -> 24 s per run):

- Iterative refinement works on the reduced embeddings (no `UMAP.transform()` per iteration)
- Edge-level consensus clustering (O(edges)) instead of all-pairs co-occurrence
- The corpus is tokenized once; the lexical TF-IDF view is derived from the keyword counts
- Batched keyword extraction; `build_hierarchy()`/`divide()` no longer re-tokenize per topic; vectorized coherence and `topic_overlap_matrix()`

**Quality**:

- Coverage-weighted c-TF-IDF (+45% NPMI on the dev splits)
- `n_topics="auto"` chooses the Leiden resolution by keyword coherence (with a guard against one-giant-topic partitions); new `resolution_` / `resolution_search_`
- Default `resolution` 1.0 -> 0.3

**Fixes**:

- kNN graph on reduced embeddings uses Euclidean distance + self-tuning kernel (`reduced_metric`); FAISS L2 distances are square-rooted
- BM25 keywords used the first *n* corpus documents instead of the topic's documents
- Metadata reweights existing edges instead of adding O(n²) clique edges
- `n_topics` / `divide()` resolution search counts only real topics and searches both directions
- NPMI coherence uses the whole corpus as reference and counts bigram keywords
- Token pattern drops numbers and underscores (`000`, `__`)
- Post-fit operations use the unrefined embedding space of the centroids

---

## Citation

```bibtex
@software{tritopic2026,
  author = {Egger, Roman},
  title = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement},
  year = {2026},
  publisher = {PyPI},
  url = {https://github.com/SmartVisions-AI/tritopic}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/SmartVisions-AI/tritopic).

## Links

- **Homepage:** [smartvisions.at](https://www.smartvisions.at)
- **Documentation:** [Full technical docs](https://github.com/SmartVisions-AI/tritopic/blob/main/docs/docs.md)
- **Repository:** [GitHub](https://github.com/SmartVisions-AI/tritopic)
- **PyPI:** [tritopic](https://pypi.org/project/tritopic/)
- **Issues:** [Bug reports & feature requests](https://github.com/SmartVisions-AI/tritopic/issues)
