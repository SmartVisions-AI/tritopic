# TriTopic

## Tri-Modal Graph Topic Modeling with Iterative Refinement and Archetypes

TriTopic is a **state-of-the-art topic modeling framework** that models topics as **stable thematic archetypes** rather than simple clusters.

By combining **semantic embeddings**, **lexical structure**, and **optional metadata** in a **robust multi-view graph**, TriTopic achieves **higher coherence**, **better stability**, and **stronger interpretability** than classical and embedding-only approaches such as BERTopic.

It is designed for **scientific reproducibility**, **multilingual corpora**, and **real-world decision support**.

---

## 🚀 Key Features

- **Tri-Modal Representation**
  - Semantic embeddings (Sentence Transformers)
  - Lexical features (TF-IDF / BM25)
  - Optional metadata integration

- **Robust Graph Construction**
  - Mutual k-Nearest Neighbors (MkNN)
  - Shared Nearest Neighbors (SNN)
  - Weighted multi-view fusion

- **Stable Topic Discovery**
  - Consensus Leiden clustering
  - Highly reproducible results across runs

- **Archetype Modeling**
  - Topics interpreted as thematic archetypes
  - Medoids + extreme archetypal documents
  - Reveals internal topic diversity

- **Iterative Refinement**
  - Topic-aware embedding optimization
  - Improves coherence and separation

- **Native Multilingual Support**
  - 60+ languages
  - Automatic language detection
  - CJK tokenization (Chinese, Japanese, Korean)

- **Optional LLM-Based Topic Labels**
  - Human-readable labels
  - Multilingual output

---

## 📦 Installation

```bash
pip install tritopic
```

Optional extras:

```bash
pip install tritopic[llm]
pip install tritopic[cjk]
pip install tritopic[full]
```

---

## ⚡ Quick Start

```python
from tritopic import TriTopic

documents = [
    "The hotel breakfast was excellent.",
    "Machine learning transforms healthcare.",
    "Climate change affects biodiversity."
]

model = TriTopic(verbose=True)
topics = model.fit_transform(documents)

model.get_topic_info()
```

---

## 🧠 Topics vs. Archetypes

In TriTopic, topics are interpreted as **archetypes**:
ideal-typical semantic structures that represent both the **core** and the **boundaries** of a theme.

Instead of selecting only “average” documents, TriTopic exposes:
- the most typical document (medoid)
- multiple archetypal extremes
- lexically dominant examples

This enables **deeper interpretation**, especially for qualitative, longitudinal, or comparative analyses.

---

## 📚 Documentation

Full methodological documentation, configuration options, and examples are available in the `/docs` folder of the repository.

---

## 📖 Citation

```bibtex
@software{tritopic2025,
  author = {Egger, Roman www.smartvisions.at},
  title = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement and Archetypes},
  year = {2026},
  url = {https://github.com/SmartVisions-AI/tritopic}
 
}
```

---

## 🏁 Summary

TriTopic moves topic modeling from **clusters** to **conceptual archetypes** —
combining robustness, interpretability, and multilingual scalability in a single framework.
