# TriTopic Documentation

## Tri-Modal Graph Topic Modeling with Iterative Refinement, Archetypes, and Multilingual Support

**Version 2.0**

TriTopic is a next-generation topic modeling framework that combines semantic, lexical, and contextual representations into a robust graph-based architecture. It introduces archetype-based topic interpretation, iterative refinement, consensus clustering, and native multilingual support.

---

## 1. Introduction

Topic modeling aims to uncover latent thematic structures in large text collections. Classical models such as LDA rely on Bag-of-Words representations and fail to capture semantic meaning. Modern embedding-based approaches improve semantic coherence but introduce instability and single-view bias.

TriTopic addresses these limitations through a multi-view, graph-based approach that treats topics as **thematic archetypes** rather than simple clusters.

---

## 2. Topics vs. Archetypes

In TriTopic, a topic is interpreted as an **archetype**:
an ideal-typical thematic structure that represents both the core and the boundaries of a theme.

Archetypes are:
- not single documents
- not statistical artifacts only
- conceptual structures around which documents organize

This distinction enables deeper qualitative interpretation and longitudinal analysis.

---

## 3. Architecture Overview

TriTopic follows a six-stage pipeline:
1. Tri-modal representation
2. Robust graph construction
3. Consensus clustering
4. Iterative refinement
5. Archetype-based representative analysis
6. Interpretation and labeling

---

## 4. Tri-Modal Representation

### 4.1 Semantic View
Sentence-transformer embeddings capture semantic similarity beyond surface wording.

### 4.2 Lexical View
TF-IDF or BM25 preserves exact term usage and domain-specific vocabulary.

### 4.3 Metadata View (Optional)
Structured metadata (source, author, time, location) is integrated as a similarity graph.

---

## 5. Robust Graph Construction

TriTopic constructs a document similarity graph using:
- Mutual k-Nearest Neighbors (MkNN)
- Shared Nearest Neighbors (SNN)

The final adjacency matrix is a weighted fusion of semantic, lexical, and metadata graphs.

---

## 6. Consensus Leiden Clustering

Leiden community detection is applied multiple times.
A co-assignment matrix is built and reclustered to ensure stability and reproducibility.

---

## 7. Iterative Refinement (Archetype Learning)

After initial clustering:
- topic centroids are computed
- document embeddings are softly pulled toward their archetype
- the graph is rebuilt and reclustered

This improves coherence, separation, and stability.

---

## 8. Archetype Analysis for Representative Documents

### 8.1 Motivation

Centroid-based representatives capture only average expressions and miss internal topic diversity.

### 8.2 Representative Types

TriTopic combines:
- **Medoid** – most typical document
- **Archetypes** – extreme boundary-defining documents
- **Keyword Champion** – lexically dominant document

### 8.3 Archetype Selection Algorithms

Supported methods:
1. **Furthest Sum** (default)
2. **Convex Hull**
3. **Principal Convex Hull Analysis (PCHA)**

---

## 9. Multilingual Support

TriTopic supports 60+ languages with:
- automatic language detection
- language-specific stopwords
- multilingual embeddings
- CJK tokenization

### 9.1 Automatic Model Selection

Embedding models are selected automatically based on detected language.

### 9.2 CJK Languages

Supported tokenizers:
- Chinese: jieba
- Japanese: fugashi / MeCab
- Korean: KoNLPy
- Thai: pythainlp

---

## 10. Configuration Reference

TriTopic exposes a comprehensive configuration interface.

```python
from tritopic import TriTopic, TriTopicConfig

config = TriTopicConfig(
    embedding_model="all-MiniLM-L6-v2",
    language="auto",
    multilingual=False,
    tokenizer="auto",
    n_neighbors=15,
    graph_type="hybrid",
    semantic_weight=0.5,
    lexical_weight=0.3,
    metadata_weight=0.2,
    n_consensus_runs=10,
    use_iterative_refinement=True,
    representative_method="hybrid",
    n_archetypes=4,
    archetype_method="furthest_sum",
    random_state=42,
    verbose=True,
)

model = TriTopic(config=config)
```

---

## 11. Usage Examples

### Example: German Text

```python
model = TriTopic(language="de")
topics = model.fit_transform(german_docs)
```

### Example: Mixed Languages

```python
model = TriTopic(multilingual=True)
topics = model.fit_transform(mixed_docs)
```

---

## 12. Scientific Positioning

TriTopic is designed for:
- archetype discovery
- multilingual corpora
- longitudinal studies
- policy and strategy analytics
- decision-support systems

The framework emphasizes reproducibility, interpretability, and transparency.

---

## 13. Citation

```bibtex
@software{tritopic2025,
  author = {Egger, Roman},
  title = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement and Archetypes},
  year = {2025},
  url = {https://github.com/roman-egger/tritopic}
}
```

---

## 14. Summary

TriTopic advances topic modeling by moving from clusters to archetypes.
It unifies multi-view representation, robust graph modeling, consensus clustering,
iterative refinement, archetype analysis, and multilingual processing in a single framework.
