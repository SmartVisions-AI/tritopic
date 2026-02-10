# TriTopic Documentation

**Version:** 2.0.0
**Author:** Roman Egger
**License:** MIT

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Evaluation & Visualization](#evaluation--visualization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

TriTopic is a state-of-the-art topic modeling library that combines semantic embeddings, lexical similarity, and optional metadata to create robust, interpretable topics through an iterative refinement process.

### Key Features

- **Multi-View Graph Fusion**: Combines semantic (SBERT), lexical (BM25), and metadata views
- **Iterative Refinement**: Automatically improves topic quality through multiple iterations
- **Consensus Clustering**: Uses multiple Leiden clustering runs for stability
- **Flexible Topic Control**: Specify exact number of topics or let the algorithm decide
- **Advanced Keyword Extraction**: c-TF-IDF, BM25, or KeyBERT methods
- **Rich Visualizations**: Interactive plots using Plotly
- **LLM-Powered Labeling**: Optional integration with Claude or GPT-4 for topic naming
- **Soft Topic Assignments**: Probabilistic topic memberships for all documents

### What Makes TriTopic Different?

Unlike traditional topic models (LDA) or simple neural approaches (Top2Vec), TriTopic:
- Combines multiple views of document similarity for robustness
- Uses graph-based clustering instead of soft clustering for clearer topic boundaries
- Iteratively refines embeddings to improve topic separation
- Provides both hard labels and soft probabilities
- Handles outliers explicitly rather than forcing all documents into topics

---

## Installation

### Requirements

- Python 3.9+
- 4GB+ RAM (8GB+ recommended)
- GPU optional (speeds up embedding generation)

### Install from PyPI

```bash
pip install tritopic
```

### Install from Source

```bash
git clone https://github.com/yourusername/tritopic.git
cd tritopic
pip install -e .
```

### Dependencies

Core dependencies are installed automatically:
- `sentence-transformers` - Embedding generation
- `scikit-learn` - Machine learning utilities
- `umap-learn` - Dimensionality reduction
- `leidenalg` - Graph clustering
- `igraph` - Graph data structures
- `plotly` - Interactive visualizations

---

## Quick Start

### Basic Usage

```python
from tritopic import TriTopic

# Your documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing handles text",
    # ... more documents
]

# Create and fit model
model = TriTopic(n_topics=5, random_state=42)
labels = model.fit_transform(documents)

# Get topic information
topic_info = model.get_topic_info()
print(topic_info)

# Visualize
fig = model.visualize()
fig.show()
```

### Working with Results

```python
# Get topics for specific documents
print(f"Document 0 is about: Topic {labels[0]}")

# Get keywords for a topic
topic = model.get_topic(topic_id=0)
print(f"Topic 0 keywords: {topic.keywords[:5]}")

# Get representative documents
rep_docs = model.get_representative_docs(topic_id=0, n_docs=3)
for idx, text in rep_docs:
    print(f"[{idx}] {text[:100]}...")

# Get topic probabilities for all documents
probs = model.probabilities_
print(f"Document 0 probabilities: {probs[0]}")
```

---

## Core Concepts

### 1. Multi-View Graph

TriTopic builds a graph where:
- **Nodes** = documents
- **Edges** = similarity between documents from multiple views:
  - **Semantic**: Embedding similarity (cosine)
  - **Lexical**: BM25 term overlap
  - **Metadata**: Optional categorical/numerical features

These views are fused into a single graph with weighted edges.

### 2. Iterative Refinement

The model alternates between:
1. **Clustering**: Find topics using Leiden algorithm
2. **Embedding**: Refine embeddings to improve within-topic similarity
3. **Convergence Check**: Stop when topic assignments stabilize (high ARI)

This process typically converges in 3-5 iterations.

### 3. Topic Assignment

- **Hard Labels**: Each document assigned to one topic (or -1 for outliers)
- **Soft Probabilities**: Probability distribution over all topics based on centroid distances
- **Outlier Detection**: Documents in very small clusters or far from centroids marked as outliers

### 4. Keyword Extraction

Topics are represented by keywords extracted using:
- **c-TF-IDF** (default): Class-based TF-IDF weighting
- **BM25**: Probabilistic term weighting
- **KeyBERT**: BERT-based keyword extraction

---

## API Reference

### TriTopic

Main class for topic modeling.

#### Constructor

```python
TriTopic(
    config: TriTopicConfig = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    n_neighbors: int = 15,
    n_topics: int | Literal["auto"] = "auto",
    use_iterative_refinement: bool = True,
    verbose: bool = True,
    random_state: int = 42
)
```

**Parameters:**
- `config` (TriTopicConfig, optional): Configuration object. If provided, overrides other parameters.
- `embedding_model` (str): Sentence-transformers model name. Popular choices:
  - `"all-MiniLM-L6-v2"` (default): Fast, good quality (384d)
  - `"all-mpnet-base-v2"`: Higher quality (768d)
  - `"paraphrase-multilingual-MiniLM-L12-v2"`: Multilingual support
- `n_neighbors` (int): Number of neighbors for graph construction. Higher = smoother topics.
- `n_topics` (int or "auto"): Target number of topics.
  - If int: Automatically reduces to this number after clustering
  - If "auto": Uses natural number from Leiden clustering
- `use_iterative_refinement` (bool): Enable iterative refinement loop
- `verbose` (bool): Print progress information
- `random_state` (int): Seed for reproducibility

**Example:**
```python
# Simple usage
model = TriTopic(n_topics=10)

# Advanced usage with custom embedding model
model = TriTopic(
    embedding_model="all-mpnet-base-v2",
    n_neighbors=20,
    n_topics=15,
    use_iterative_refinement=True,
    random_state=42
)
```

#### Methods

##### fit()

```python
fit(
    documents: list[str],
    embeddings: np.ndarray = None,
    metadata: pd.DataFrame = None
) -> TriTopic
```

Fit the topic model to documents.

**Parameters:**
- `documents`: List of document texts
- `embeddings` (optional): Pre-computed embeddings (shape: [n_docs, embedding_dim])
- `metadata` (optional): DataFrame with document metadata (same length as documents)

**Returns:** Self (fitted model)

**Example:**
```python
model.fit(documents)

# Or with pre-computed embeddings
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(documents)
model.fit(documents, embeddings=embeddings)
```

##### fit_transform()

```python
fit_transform(
    documents: list[str],
    embeddings: np.ndarray = None,
    metadata: pd.DataFrame = None
) -> np.ndarray
```

Fit the model and return topic assignments.

**Returns:** Array of topic labels (shape: [n_docs])

**Example:**
```python
labels = model.fit_transform(documents)
print(f"Found {len(set(labels[labels >= 0]))} topics")
```

##### transform()

```python
transform(documents: list[str]) -> np.ndarray
```

Assign topics to new documents.

**Parameters:**
- `documents`: List of new document texts

**Returns:** Array of topic labels

**Example:**
```python
# Fit on training data
model.fit(train_documents)

# Predict on new documents
new_labels = model.transform(test_documents)
```

##### transform_proba()

```python
transform_proba(documents: list[str]) -> np.ndarray
```

Get topic probability distributions for new documents.

**Returns:** Array of probabilities (shape: [n_docs, n_topics])

**Example:**
```python
probs = model.transform_proba(["New document about machine learning"])
print(f"Most likely topic: {probs.argmax()}")
print(f"Confidence: {probs.max():.2%}")
```

##### get_topic_info()

```python
get_topic_info() -> pd.DataFrame
```

Get a DataFrame with all topic information.

**Returns:** DataFrame with columns:
- `Topic`: Topic ID
- `Size`: Number of documents
- `Keywords`: Top 5 keywords (comma-separated)
- `All_Keywords`: All keywords (list)
- `Label`: Human-readable label (if generated)
- `Description`: Topic description (if generated)
- `Coherence`: NPMI coherence score

**Example:**
```python
topic_df = model.get_topic_info()
print(topic_df[["Topic", "Size", "Keywords", "Coherence"]])
```

##### get_topic()

```python
get_topic(topic_id: int) -> TopicInfo | None
```

Get detailed information about a specific topic.

**Returns:** TopicInfo object with attributes:
- `topic_id`: Topic identifier
- `size`: Number of documents
- `keywords`: List of keyword strings
- `keyword_scores`: Keyword importance scores
- `representative_docs`: Indices of representative documents
- `centroid`: Topic centroid embedding
- `coherence`: NPMI coherence score
- `label`: Human-readable label
- `description`: Topic description

**Example:**
```python
topic = model.get_topic(0)
print(f"Topic {topic.topic_id}: {', '.join(topic.keywords[:5])}")
print(f"Size: {topic.size} documents")
print(f"Coherence: {topic.coherence:.3f}")
```

##### get_representative_docs()

```python
get_representative_docs(
    topic_id: int,
    n_docs: int = 5
) -> list[tuple[int, str]]
```

Get the most representative documents for a topic.

**Returns:** List of (index, text) tuples

**Example:**
```python
rep_docs = model.get_representative_docs(topic_id=0, n_docs=3)
for idx, text in rep_docs:
    print(f"[{idx}] {text[:200]}...")
```

##### reduce_topics()

```python
reduce_topics(n_topics: int) -> TriTopic
```

Reduce the number of topics by merging similar ones.

**Parameters:**
- `n_topics`: Target number of topics

**Returns:** Self (modified model)

**Note:** This method modifies the model in-place and updates all attributes (labels, probabilities, topics, etc.)

**Example:**
```python
# Fit with auto mode
model = TriTopic(n_topics="auto")
model.fit(documents)
print(f"Found {len(model.topics_)} topics")

# Reduce to 10 topics
model.reduce_topics(10)
print(f"Now have {len(model.topics_)} topics")
```

##### merge_topics()

```python
merge_topics(topic_ids: list[int]) -> TriTopic
```

Merge specific topics together.

**Parameters:**
- `topic_ids`: List of topic IDs to merge

**Returns:** Self (modified model)

**Example:**
```python
# Merge topics 5, 7, and 12
model.merge_topics([5, 7, 12])
```

##### reduce_outliers()

```python
reduce_outliers(
    strategy: Literal["embeddings", "neighbors"] = "embeddings",
    threshold: float = 0.1
) -> TriTopic
```

Reassign outlier documents to topics.

**Parameters:**
- `strategy`:
  - `"embeddings"`: Assign to nearest centroid if similarity > threshold
  - `"neighbors"`: Assign based on k-nearest neighbors voting
- `threshold`: Minimum similarity for embedding strategy

**Returns:** Self (modified model)

**Example:**
```python
# Check outliers
print(f"Outliers: {(model.labels_ == -1).sum()}")

# Reduce using embeddings
model.reduce_outliers(strategy="embeddings", threshold=0.1)
print(f"Outliers after: {(model.labels_ == -1).sum()}")

# Reduce remaining using neighbors
model.reduce_outliers(strategy="neighbors")
```

##### generate_labels()

```python
generate_labels(labeler) -> TriTopic
```

Generate human-readable labels for topics using an LLM or simple method.

**Parameters:**
- `labeler`: Instance of `LLMLabeler` or `SimpleLabeler`

**Returns:** Self (modified model)

**Example:**
```python
from tritopic import SimpleLabeler, LLMLabeler

# Simple labeling (concatenate top keywords)
simple_labeler = SimpleLabeler(n_words=3)
model.generate_labels(simple_labeler)

# LLM labeling (requires API key)
llm_labeler = LLMLabeler(
    provider="anthropic",
    api_key="your-api-key",
    model="claude-3-haiku-20240307",
    language="english",
    domain_hint="news articles"
)
model.generate_labels(llm_labeler)

# View labels
topic_df = model.get_topic_info()
print(topic_df[["Topic", "Label", "Description"]])
```

##### evaluate()

```python
evaluate() -> dict
```

Compute evaluation metrics for the topic model.

**Returns:** Dictionary with metrics:
- `coherence_mean`: Mean NPMI coherence across topics
- `coherence_std`: Standard deviation of coherence
- `diversity`: Percentage of unique keywords across topics
- `stability`: Cluster stability (if multiple runs)
- `n_topics`: Number of topics
- `outlier_ratio`: Fraction of outlier documents

**Example:**
```python
metrics = model.evaluate()
print(f"Coherence: {metrics['coherence_mean']:.3f}")
print(f"Diversity: {metrics['diversity']:.3f}")
print(f"Outliers: {metrics['outlier_ratio']:.1%}")
```

##### visualize()

```python
visualize(
    method: str = "umap",
    show_outliers: bool = True,
    title: str = None
) -> plotly.graph_objs.Figure
```

Create an interactive 2D visualization of documents colored by topic.

**Parameters:**
- `method`: Dimensionality reduction method ("umap" or "tsne")
- `show_outliers`: Whether to show outlier documents
- `title`: Custom plot title

**Returns:** Plotly Figure object

**Example:**
```python
fig = model.visualize(method="umap", title="My Topics")
fig.show()
fig.write_html("topics.html")
```

##### visualize_topics()

```python
visualize_topics(
    n_keywords: int = 10,
    title: str = None
) -> plotly.graph_objs.Figure
```

Visualize topic keywords as horizontal bar charts.

**Returns:** Plotly Figure object

**Example:**
```python
fig = model.visualize_topics(n_keywords=8)
fig.show()
```

##### visualize_hierarchy()

```python
visualize_hierarchy(title: str = None) -> plotly.graph_objs.Figure
```

Show hierarchical relationships between topics using a dendrogram.

**Returns:** Plotly Figure object

**Example:**
```python
fig = model.visualize_hierarchy()
fig.show()
```

##### save() / load()

```python
save(path: str) -> None
load(path: str) -> TriTopic  # class method
```

Save and load fitted models.

**Example:**
```python
# Save
model.save("my_model.pkl")

# Load
loaded_model = TriTopic.load("my_model.pkl")
```

#### Attributes (After Fitting)

- `labels_` (np.ndarray): Topic assignments for each document
- `embeddings_` (np.ndarray): Document embeddings
- `reduced_embeddings_` (np.ndarray): UMAP-reduced embeddings
- `probabilities_` (np.ndarray): Topic probability matrix
- `topics_` (list[TopicInfo]): List of TopicInfo objects
- `topic_embeddings_` (np.ndarray): Topic centroid embeddings
- `documents_` (list[str]): Original documents (if stored)

---

## Configuration

### TriTopicConfig

For advanced control, create a custom configuration:

```python
from tritopic import TriTopicConfig, TriTopic

config = TriTopicConfig(
    # Embedding settings
    embedding_model="all-mpnet-base-v2",
    embedding_batch_size=64,

    # Graph settings
    n_neighbors=20,
    metric="cosine",
    graph_type="hybrid",  # "mutual_knn", "snn", or "hybrid"
    snn_weight=0.5,

    # Multi-view settings
    use_lexical_view=True,
    semantic_weight=0.6,
    lexical_weight=0.4,

    # Clustering settings
    resolution=1.0,
    n_consensus_runs=15,
    min_cluster_size=10,

    # Iterative refinement
    use_iterative_refinement=True,
    max_iterations=5,
    convergence_threshold=0.95,

    # Keywords
    keyword_method="ctfidf",  # "ctfidf", "bm25", or "keybert"
    n_keywords=15,
    n_representative_docs=5,

    # Dimensionality reduction
    use_dim_reduction=True,
    reduced_dims=10,
    dim_reduction_method="umap",

    # Misc
    random_state=42,
    verbose=True
)

model = TriTopic(config=config)
```

### Key Configuration Parameters

#### Embedding Settings
- `embedding_model`: Sentence-transformer model name
- `embedding_batch_size`: Batch size for encoding (tune based on GPU memory)

#### Graph Settings
- `n_neighbors`: Higher = smoother topics, lower = more fine-grained
- `graph_type`:
  - `"mutual_knn"`: Only bidirectional nearest neighbors
  - `"snn"`: Shared nearest neighbor graph
  - `"hybrid"`: Combination of both (recommended)

#### Multi-View Settings
- `use_lexical_view`: Enable BM25 lexical similarity
- `semantic_weight`: Weight for embedding similarity
- `lexical_weight`: Weight for BM25 similarity

#### Clustering Settings
- `resolution`: Leiden resolution parameter (higher = more topics)
- `n_consensus_runs`: More runs = more stable but slower
- `min_cluster_size`: Smaller clusters become outliers

#### Iterative Refinement
- `max_iterations`: Maximum refinement iterations
- `convergence_threshold`: ARI threshold to stop (0.95 = 95% agreement)

---

## Advanced Usage

### Custom Embeddings

```python
from sentence_transformers import SentenceTransformer

# Use a different encoder
encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = encoder.encode(documents, show_progress_bar=True)

# Fit with pre-computed embeddings
model = TriTopic(n_topics=10)
model.fit(documents, embeddings=embeddings)
```

### Incremental Topic Discovery

```python
# Start with auto mode
model = TriTopic(n_topics="auto", verbose=True)
model.fit(documents)
print(f"Found {len(model.topics_)} topics naturally")

# If too many topics, reduce
if len(model.topics_) > 20:
    model.reduce_topics(20)

# Or merge specific similar topics
model.merge_topics([5, 12])  # Merge topics 5 and 12
```

### Working with Large Datasets

```python
# Use lighter embedding model
model = TriTopic(
    embedding_model="all-MiniLM-L6-v2",  # Faster, smaller
    embedding_batch_size=128,  # Larger batches if GPU available
    use_dim_reduction=True,  # Reduce dimensionality for speed
    reduced_dims=10
)

# Disable iterative refinement for speed
model = TriTopic(
    n_topics=20,
    use_iterative_refinement=False  # Single-pass mode
)
```

### Multilingual Documents

```python
# Use multilingual model
model = TriTopic(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)
model.fit(multilingual_documents)
```

### Fine-tuning Topic Granularity

```python
from tritopic import TriTopicConfig

# More fine-grained topics
config = TriTopicConfig(
    resolution=1.5,  # Higher resolution
    min_cluster_size=5,  # Smaller minimum cluster size
    n_neighbors=10  # Fewer neighbors
)

# Broader, more general topics
config = TriTopicConfig(
    resolution=0.7,  # Lower resolution
    min_cluster_size=20,  # Larger minimum cluster size
    n_neighbors=30  # More neighbors
)
```

---

## Evaluation & Visualization

### Intrinsic Metrics

```python
metrics = model.evaluate()

print(f"Coherence (NPMI): {metrics['coherence_mean']:.3f} ± {metrics['coherence_std']:.3f}")
print(f"Diversity: {metrics['diversity']:.3f}")
print(f"Number of topics: {metrics['n_topics']}")
print(f"Outlier ratio: {metrics['outlier_ratio']:.1%}")
```

### External Metrics (with ground truth)

```python
from tritopic.utils.metrics import compute_downstream_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Assuming you have true_labels
ari = adjusted_rand_score(true_labels, model.labels_)
nmi = normalized_mutual_info_score(true_labels, model.labels_)

print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")

# Downstream classification performance
f1 = compute_downstream_score(
    model.embeddings_,
    model.labels_,
    true_labels,
    task="classification"
)
print(f"Downstream F1: {f1:.3f}")
```

### Visualizations

```python
# 1. Document map (2D projection)
fig = model.visualize(method="umap", title="Document Clustering")
fig.show()

# 2. Topic keywords
fig = model.visualize_topics(n_keywords=10)
fig.show()

# 3. Topic hierarchy
fig = model.visualize_hierarchy()
fig.show()

# 4. Topic similarity heatmap
from tritopic import TopicVisualizer
viz = TopicVisualizer()
fig = viz.plot_topic_similarity(
    model.topic_embeddings_,
    model.topics_,
    title="Topic Similarity Matrix"
)
fig.show()

# 5. Topics over time (if you have timestamps)
import pandas as pd
timestamps = pd.date_range("2024-01-01", periods=len(documents), freq="D")
fig = viz.plot_topic_over_time(
    labels=model.labels_,
    timestamps=timestamps,
    topics=model.topics_
)
fig.show()
```

---

## Best Practices

### 1. Choosing n_topics

```python
# Strategy 1: Let algorithm decide
model = TriTopic(n_topics="auto")
model.fit(documents)
print(f"Natural clustering found {len(model.topics_)} topics")

# Strategy 2: Try multiple values
for n in [10, 20, 30]:
    model = TriTopic(n_topics=n, random_state=42)
    model.fit(documents)
    metrics = model.evaluate()
    print(f"n={n}: Coherence={metrics['coherence_mean']:.3f}, "
          f"Diversity={metrics['diversity']:.3f}")
```

### 2. Handling Outliers

```python
# Check outlier percentage
outlier_pct = (model.labels_ == -1).mean()
print(f"Outliers: {outlier_pct:.1%}")

# If too many outliers (>10%), try:
# Option 1: Reduce outliers
model.reduce_outliers(strategy="embeddings", threshold=0.05)

# Option 2: Adjust clustering parameters
config = TriTopicConfig(min_cluster_size=5)  # Smaller clusters
model = TriTopic(config=config)
```

### 3. Improving Topic Quality

```python
# Use higher quality embeddings
model = TriTopic(
    embedding_model="all-mpnet-base-v2",  # Better than MiniLM
    n_keywords=15  # More keywords per topic
)

# Enable iterative refinement
model = TriTopic(
    use_iterative_refinement=True,
    max_iterations=7,  # More iterations
    convergence_threshold=0.97  # Stricter convergence
)

# Increase consensus runs
config = TriTopicConfig(n_consensus_runs=20)
model = TriTopic(config=config)
```

### 4. Topic Naming

```python
# For quick exploration
from tritopic import SimpleLabeler
model.generate_labels(SimpleLabeler(n_words=3))

# For production/presentation
from tritopic import LLMLabeler
labeler = LLMLabeler(
    provider="anthropic",
    api_key="your-api-key",
    model="claude-3-haiku-20240307",
    language="english",
    domain_hint="scientific papers about machine learning"
)
model.generate_labels(labeler)
```

### 5. Document Preprocessing

```python
# Remove very short documents
documents = [doc for doc in documents if len(doc.split()) > 10]

# Remove duplicates
documents = list(set(documents))

# Handle special characters if needed
import re
documents = [re.sub(r'[^\w\s]', ' ', doc) for doc in documents]
documents = [' '.join(doc.split()) for doc in documents]  # Normalize whitespace
```

---

## Troubleshooting

### Issue: Model finds too many topics

**Solutions:**
```python
# Lower resolution
config = TriTopicConfig(resolution=0.7)

# Increase minimum cluster size
config = TriTopicConfig(min_cluster_size=20)

# Or explicitly set n_topics
model = TriTopic(n_topics=15)
```

### Issue: Model finds too few topics

**Solutions:**
```python
# Increase resolution
config = TriTopicConfig(resolution=1.5)

# Decrease minimum cluster size
config = TriTopicConfig(min_cluster_size=5)

# Reduce n_neighbors for finer granularity
model = TriTopic(n_neighbors=10)
```

### Issue: Low coherence scores

**Solutions:**
```python
# Use better embedding model
model = TriTopic(embedding_model="all-mpnet-base-v2")

# Enable iterative refinement
model = TriTopic(use_iterative_refinement=True)

# Try different keyword extraction method
config = TriTopicConfig(keyword_method="keybert")
```

### Issue: Many outliers

**Solutions:**
```python
# Reduce outliers after fitting
model.reduce_outliers(strategy="embeddings", threshold=0.1)
model.reduce_outliers(strategy="neighbors")

# Or adjust clustering
config = TriTopicConfig(
    min_cluster_size=5,  # Smaller clusters
    outlier_threshold=0.05  # Lower threshold
)
```

### Issue: Out of memory

**Solutions:**
```python
# Use lighter embedding model
model = TriTopic(embedding_model="all-MiniLM-L6-v2")

# Reduce batch size
config = TriTopicConfig(embedding_batch_size=16)

# Disable lexical view (uses less memory)
config = TriTopicConfig(use_lexical_view=False)

# Use dimensionality reduction
config = TriTopicConfig(
    use_dim_reduction=True,
    reduced_dims=10
)
```

### Issue: Slow performance

**Solutions:**
```python
# Disable iterative refinement
model = TriTopic(use_iterative_refinement=False)

# Reduce consensus runs
config = TriTopicConfig(n_consensus_runs=5)

# Use GPU for embeddings (automatic if available)

# Sample documents for large datasets
import random
sample = random.sample(documents, 10000)
model.fit(sample)
```

---

## Examples

See the `examples/` folder for complete notebooks:
- `tritopic_full_demo.ipynb` - Complete feature demonstration
- Additional examples for specific use cases

---

## Citation

If you use TriTopic in your research, please cite:

```bibtex
@software{tritopic2024,
  author = {Egger, Roman},
  title = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement},
  year = {2024},
  url = {https://github.com/yourusername/tritopic}
}
```

---

## Support

- **Issues**: https://github.com/yourusername/tritopic/issues
- **Discussions**: https://github.com/yourusername/tritopic/discussions
- **Email**: your.email@example.com

---

## License

MIT License - see LICENSE file for details.
