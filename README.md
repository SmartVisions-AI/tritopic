# TriTopic

**Tri-Modal Graph Topic Modeling with Iterative Refinement**

A topic modeling library that fuses semantic embeddings, lexical similarity, and metadata context through multi-view graph construction, consensus Leiden clustering, and iterative refinement to produce stable, interpretable topics.

[![PyPI version](https://badge.fury.io/py/tritopic.svg)](https://badge.fury.io/py/tritopic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Pipeline](#the-pipeline)
- [Configuration Reference](#configuration-reference)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Soft Topic Assignments](#soft-topic-assignments)
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
- [Citation](#citation)
- [License](#license)

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-view graph fusion** | Combines semantic embeddings, TF-IDF lexical similarity, and optional metadata into a single graph, avoiding the "embedding blur" that single-view models suffer from |
| **Mutual kNN + SNN graphs** | Eliminates noise bridges between unrelated documents using bidirectional neighbor checks and shared-neighbor weighting |
| **Consensus Leiden clustering** | Runs the Leiden algorithm multiple times and merges results via a co-occurrence matrix, producing dramatically more stable topics than single-run approaches |
| **Iterative refinement** | Alternates between clustering and embedding refinement, pulling documents toward their topic centroids to sharpen boundaries |
| **Dimensionality reduction** | Reduces high-dimensional embeddings (384-768d) to ~10d with UMAP or PaCMAP before graph construction, improving neighbor quality |
| **Soft topic assignments** | Computes per-document probability distributions over all topics, not just hard labels |
| **Post-fit outlier reduction** | Reassigns outlier documents using centroid similarity or neighbor voting after the model is fitted |
| **Hierarchical topic merging** | Iteratively merges the most similar topic pairs to reach a target count, or manually merges specific topics |
| **LLM-powered labels** | Generates human-readable topic names via Claude or GPT-4 |
| **Interactive visualizations** | 2D document maps, keyword bar charts, dendrograms, similarity heatmaps, and temporal topic evolution via Plotly |

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

**Optional:** anthropic, openai (for LLM labeling), pacmap, datamapplot

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

**Step 3 - Metadata graph (optional):** Categorical and numerical metadata fields create additional edges between related documents.

**Step 4 - Multi-view graph fusion:** The semantic kNN graph (built on reduced embeddings), lexical graph, and metadata graph are combined with configurable weights into a single igraph Graph. The semantic graph can use mutual kNN, SNN, or a hybrid of both.

**Step 5 - Consensus Leiden clustering:** The Leiden algorithm runs multiple times (default: 10) with different seeds. A co-occurrence matrix records how often each pair of documents lands in the same cluster. Hierarchical clustering on this matrix produces a consensus partition that is far more stable than any single run. Clusters below `min_cluster_size` are marked as outliers (-1).

**Step 6 - Iterative refinement:** Embeddings are softly blended toward their topic centroid (20% pull), then the graph and clustering are re-run. This loop continues until the Adjusted Rand Index between consecutive iterations exceeds the convergence threshold (default: 0.95), or until `max_iterations` is reached. During iterative refinement, the dimensionality reducer transforms the refined embeddings for each new graph-building pass.

**Step 7 - Keywords and centroids:** c-TF-IDF (or BM25/KeyBERT) extracts representative keywords per topic. Topic centroids are computed as the mean embedding of each topic's documents. Soft probabilities are computed via cosine similarity to centroids passed through softmax.

---

## Configuration Reference

All parameters are set through `TriTopicConfig` or as constructor overrides:

```python
from tritopic import TriTopic, TriTopicConfig

config = TriTopicConfig(
    # --- Embedding ---
    embedding_model="all-MiniLM-L6-v2",   # sentence-transformers model name
    embedding_batch_size=32,               # encoding batch size

    # --- Dimensionality Reduction ---
    use_dim_reduction=True,                # reduce before graph building
    reduced_dims=10,                       # target dimensionality
    dim_reduction_method="umap",           # "umap" or "pacmap"
    umap_n_neighbors=15,                   # UMAP/PaCMAP neighbor count
    umap_min_dist=0.0,                     # 0.0 optimized for clustering

    # --- Graph Construction ---
    n_neighbors=15,                        # k for kNN graph
    metric="cosine",                       # distance metric
    graph_type="hybrid",                   # "knn", "mutual_knn", "snn", "hybrid"
    snn_weight=0.5,                        # SNN weight in hybrid mode

    # --- Multi-View Fusion ---
    use_lexical_view=True,                 # include TF-IDF view
    use_metadata_view=False,               # include metadata view
    semantic_weight=0.5,                   # weight for semantic graph
    lexical_weight=0.3,                    # weight for lexical graph
    metadata_weight=0.2,                   # weight for metadata graph

    # --- Clustering ---
    resolution=1.0,                        # Leiden resolution (higher = more topics)
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

    # --- Outlier Handling ---
    outlier_threshold=0.1,                 # cosine similarity threshold for transform()

    # --- Misc ---
    random_state=42,
    verbose=True,
)

model = TriTopic(config=config)
```

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
print(model.embeddings_.shape)          # (n_docs, 384)  full embeddings kept
```

**How it works:**

- Reduced embeddings are used only for graph construction (kNN neighbor search)
- Full-dimensional embeddings are used for centroid computation, keyword extraction, representative docs, and similarity calculations
- During iterative refinement, refined embeddings are re-projected through the fitted reducer at each iteration
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

---

## Outlier Reduction

Leiden clustering combined with small-cluster removal can produce 20-40% outliers. `reduce_outliers()` reassigns them post-fit.

### Strategy: embeddings (default)

Each outlier is assigned to the topic whose centroid is most similar, if the similarity exceeds a threshold:

```python
model.fit(documents)
print(f"Outliers before: {(model.labels_ == -1).sum()}")

# Default threshold = config.outlier_threshold (0.1)
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

At each step, the two most similar centroids are found, and the smaller topic is relabeled to the larger one. After all merges complete, keywords and centroids are re-extracted from scratch.

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

Class-based TF-IDF treats all documents in a topic as a single "class document" and scores terms by their distinctiveness for that topic compared to the corpus. This is the same approach used by BERTopic.

```python
model.config.keyword_method = "ctfidf"
```

### BM25

BM25 scoring is more robust to document length variations than TF-IDF:

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
    model="claude-3-haiku-20240307",   # fast and cheap
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
| `coherence_mean` | -1 to 1 | Average NPMI coherence across topics (higher = more coherent keywords) |
| `coherence_std` | 0+ | Standard deviation of coherence across topics |
| `diversity` | 0 to 1 | Proportion of unique keywords across all topics (higher = more distinct topics) |
| `stability` | -1 to 1 | Average pairwise ARI across consensus runs (higher = more reproducible) |
| `n_topics` | 1+ | Number of non-outlier topics |
| `outlier_ratio` | 0 to 1 | Fraction of documents labeled as outliers |

Additional metrics are available as standalone functions:

```python
from tritopic.utils.metrics import (
    compute_coherence,
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

Documents with shared metadata (source, category, date) get additional graph edges:

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

Categorical columns create edges between documents with matching values. Numerical columns create edges between documents with similar values (similarity > 0.8 after normalization).

### Finding the optimal resolution

The resolution parameter controls how many topics Leiden produces. You can search for the best value:

```python
from tritopic.core.clustering import ConsensusLeiden

# After initial fit
clusterer = ConsensusLeiden()
optimal = clusterer.find_optimal_resolution(
    graph=model.graph_,
    resolution_range=(0.5, 2.0),
    n_steps=10,
    target_n_topics=15,       # optional: aim for ~15 topics
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
| `visualize(method?, show_outliers?, ...)` | 2D document scatter plot. |
| `visualize_topics(n_keywords?, ...)` | Keyword bar charts per topic. |
| `visualize_hierarchy(...)` | Topic dendrogram. |
| `save(path)` | Pickle model to disk (includes all state, reducer, probabilities). |
| `TriTopic.load(path)` | Class method to load a saved model. |

### Key attributes after fit

| Attribute | Type | Description |
|---|---|---|
| `labels_` | `np.ndarray` | Hard topic assignment per document. -1 = outlier. |
| `probabilities_` | `np.ndarray` | Soft assignments, shape `(n_docs, n_topics)`. Rows sum to ~1. |
| `embeddings_` | `np.ndarray` | Full-dimensional document embeddings (refined if iterative). |
| `reduced_embeddings_` | `np.ndarray` | Low-dimensional embeddings used for graph building. |
| `topic_embeddings_` | `np.ndarray` | Centroid embedding per topic, shape `(n_topics, embed_dim)`. |
| `topics_` | `list[TopicInfo]` | List of `TopicInfo` objects with keywords, scores, centroids. |
| `documents_` | `list[str]` | Stored training documents. |
| `graph_` | `igraph.Graph` | The final fused graph. |

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

Running Leiden once is sensitive to random initialization. TriTopic runs it `n_consensus_runs` times (default: 10) with different seeds and builds a co-occurrence matrix recording how often each document pair was assigned to the same cluster. Hierarchical clustering (average linkage) on this matrix produces the final partition, selected by maximizing the average ARI with all individual runs. The stability score (average pairwise ARI across runs) quantifies how reproducible the clustering is.

### Iterative refinement

After an initial clustering pass, document embeddings are softly blended toward their topic centroid: `refined = 0.8 * original + 0.2 * centroid`, then L2-normalized. The full pipeline (graph + clustering) re-runs on the refined embeddings. This process converges when consecutive partitions have ARI >= 0.95 (configurable). The effect is tighter, more separated topic clusters.

### Supported embedding models

Any model from the [sentence-transformers](https://www.sbert.net/) library works. Recommended choices:

| Model | Dimensions | Speed | Quality | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default. Best speed/quality tradeoff. |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality, 2x slower. |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Best | State-of-the-art for English. |
| `BAAI/bge-m3` | 1024 | Slow | Best | Multilingual support. |
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
| **Soft assignments** | Via HDBSCAN probabilities | Cosine similarity + softmax |
| **Outlier reduction** | 4 strategies | 2 strategies (embeddings, neighbors) |
| **Topic merging** | Hierarchical | Hierarchical + manual merge |
| **Keyword extraction** | c-TF-IDF | c-TF-IDF, BM25, or KeyBERT |
| **LLM labels** | Via representation model | Built-in Claude/GPT-4 support |

---

## Citation

```bibtex
@software{tritopic2025,
  author = {Egger, Roman},
  title = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement},
  year = {2025},
  url = {https://github.com/SmartVisions-AI/tritopic}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please open an issue or pull request on [GitHub](https://github.com/SmartVisions-AI/tritopic).
