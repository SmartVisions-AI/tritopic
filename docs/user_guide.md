# TriTopic User Guide

Task-by-task instructions with code. Every snippet comes from a runnable script in
[`examples/`](../examples); the [technical documentation](docs.md) explains how each part works and why.

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Your own data](#3-your-own-data)
4. [How many topics?](#4-how-many-topics)
5. [Shaping the topics after fitting](#5-shaping-the-topics-after-fitting)
6. [New documents, saving and loading](#6-new-documents-saving-and-loading)
7. [Metadata, time and languages](#7-metadata-time-and-languages)
8. [Let an LLM interpret the topics](#8-let-an-llm-interpret-the-topics)
9. [Typed LLM judgements (Decisions API)](#9-typed-llm-judgements-decisions-api)
10. [Start from a codebook (seeded topics)](#10-start-from-a-codebook-seeded-topics)
11. [Research toolkit](#11-research-toolkit)
12. [Codebook and second coder for manual coding](#12-codebook-and-second-coder-for-manual-coding)
13. [Evaluation and visualization](#13-evaluation-and-visualization)
14. [Configuration recipes](#14-configuration-recipes)
15. [FAQ and troubleshooting](#15-faq-and-troubleshooting)

---

## 1. Installation

```bash
pip install git+https://github.com/SmartVisions-AI/tritopic.git     # 2.5 (this repository)
pip install "tritopic[benchmark] @ git+https://github.com/SmartVisions-AI/tritopic.git"  # + BERTopic, datasets
```

Python 3.9-3.13. The first fit downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB).

The LLM features (`TopicInterpreter`, the Decisions API functions, `codebook()`, `intercoder_reliability()`)
call the OpenAI API over HTTPS and only need an API key:

```bash
export OPENAI_API_KEY=sk-...          # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

## 2. Quick start

Script: [`examples/01_quickstart.py`](../examples/01_quickstart.py)

```python
from tritopic import TriTopic

model = TriTopic()                          # n_topics="auto"
labels = model.fit_transform(documents)     # one topic id per document

print(model.get_topic_info())               # Topic, Size, Keywords, Label, Coherence, ...
topic = model.get_topic(0)
print(topic.keywords, topic.size)
print(model.get_representative_docs(0, n_docs=3))
model.visualize().write_html("map.html")    # interactive 2D map
```

`documents` is a list of strings. Every document gets a topic; `-1` (outlier) only appears for clusters
smaller than `min_cluster_size`.

## 3. Your own data

Script: [`examples/02_your_own_data.py`](../examples/02_your_own_data.py)

```python
import re
import pandas as pd
from tritopic import TriTopic, TriTopicConfig

df = pd.read_csv("my_texts.csv")
df["text"] = df["text"].fillna("").map(lambda t: re.sub(r"\s+", " ", t).strip())
df = df[df["text"].str.split().str.len() >= 10].reset_index(drop=True)   # drop very short texts
documents = df["text"].tolist()

# Optional: your own embeddings (any sentence-transformers model, GPU, caching ...)
from sentence_transformers import SentenceTransformer
embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(documents, normalize_embeddings=True)

config = TriTopicConfig(min_cluster_size=10, n_keywords=10, random_state=42, verbose=False)
model = TriTopic(config=config, n_topics=8)
df["topic"] = model.fit_transform(documents, embeddings=embeddings)

info = model.get_topic_info().set_index("Topic")
df["topic_keywords"] = df["topic"].map(info["Keywords"])
df["topic_probability"] = model.probabilities_.max(axis=1)
df.to_csv("my_texts_with_topics.csv", index=False)
```

Tips:

- Clean only what is noise for your question (HTML, signatures, boilerplate). Stop words are handled by
  `language`; do not stem or lemmatize, the embeddings need natural text.
- Long documents are fine. Embedding models truncate (MiniLM: 256 tokens), so for books or long reports
  split into paragraphs or sections first and fit on the parts.
- Pre-computed embeddings make experiments fast: encode once, fit many times.

## 4. How many topics?

Script: [`examples/03_shape_the_topics.py`](../examples/03_shape_the_topics.py)

```python
TriTopic()                                            # "auto": coarsest partition with near-best keyword coherence
TriTopic(n_topics=12)                                 # exactly 12 topics
TriTopic(config=TriTopicConfig(auto_resolution=False, resolution=0.5))   # fixed Leiden resolution

model.resolution_                                     # chosen resolution
model.resolution_search_                              # [(resolution, n_topics, coherence), ...]
```

Auto mode tends to give more topics than a corpus has classes (BBC: about 6-14 for 5 sections). If you
know the granularity you need, pass `n_topics`; if you know the categories, use seeds (§10).

## 5. Shaping the topics after fitting

```python
model.reduce_outliers(strategy="neighbors")      # or "embeddings", or "decisions" (LLM, §9)
model.reduce_topics(10)                          # merge the most similar topics until 10 remain
model.merge_topics([4, 7])                       # merge specific topics
model.divide(topic_id=0, n_subtopics=3)          # split one topic

hierarchy = model.build_hierarchy(n_levels=3)    # coarse -> fine
for root in hierarchy.roots:
    print(root.node_id, root.size, root.keywords[:4], [c.node_id for c in root.children])
model.visualize_hierarchy_tree().write_html("hierarchy.html")
```

All operations refresh keywords, centroids and probabilities.

## 6. New documents, saving and loading

Script: [`examples/04_new_documents_and_saving.py`](../examples/04_new_documents_and_saving.py)

```python
model.transform(new_docs)            # topic ids; -1 if less similar than model.outlier_threshold_
model.transform_proba(new_docs)      # (n_new, n_topics), columns in topic-id order
model.probabilities_                 # the same for the training documents

model.save("model.pkl")
model = TriTopic.load("model.pkl")   # keeps topics, labels, seeds, reducer, embeddings
```

The outlier threshold is calibrated on the training data (1st percentile of the training documents'
similarity to their own topic). Set `TriTopicConfig(outlier_threshold=0.3)` for a fixed value.

## 7. Metadata, time and languages

Script: [`examples/05_metadata_time_and_languages.py`](../examples/05_metadata_time_and_languages.py)

```python
config = TriTopicConfig(use_metadata_view=True, metadata_weight=0.2)
model = TriTopic(config=config).fit(documents, metadata=df[["source", "rating"]])
pd.crosstab(df["source"], model.labels_, normalize="index")       # topic share per source

from tritopic import TopicVisualizer
TopicVisualizer().plot_topic_over_time(labels=model.labels_, timestamps=df["date"].tolist(),
                                       topics=model.topics_).write_html("over_time.html")

TriTopic(language="german")                        # german / french / spanish stop words
TriTopic(language="multilingual")                  # switches to BAAI/bge-m3 (100+ languages)
```

Metadata only strengthens edges between documents that are already similar in content; it never connects
unrelated documents. For how topics change content over time, see `topic_evolution()` in §11.

## 8. Let an LLM interpret the topics

Script: [`examples/06_llm_interpretation.py`](../examples/06_llm_interpretation.py)

```python
from tritopic import TopicInterpreter

interpreter = TopicInterpreter(domain_hint="customer reviews of hotels", language="English")
results = interpreter.interpret(model)          # writes label + description into model.topics_
for tid, r in results.items():
    print(tid, r.label, r.verdict, r.description, r.aspects)

log = interpreter.refine(model, results=results)   # split topics the LLM calls "mixed"
print(interpreter.summarize(model))                # overview of the topic landscape
```

`refine()` only keeps a split when the graph split agrees with the LLM's sub-themes. Default model
`gpt-6-luna`; pass `model="..."` to use another. Cost: about 1,500 input tokens per topic.

Labels only, with Anthropic or OpenAI SDKs: `model.generate_labels(LLMLabeler(provider="anthropic", api_key=...))`.
Without any LLM: `model.generate_labels(SimpleLabeler(n_words=3))`.

## 9. Typed LLM judgements (Decisions API)

Script: [`examples/07_decisions_api.py`](../examples/07_decisions_api.py)

```python
from tritopic.integrations.decisions import (DecisionsClient, word_intrusion, rate_topics,
                                             assign_documents, suggest_merges, apply_merges)

client = DecisionsClient()
keywords = [t.keywords for t in model.topics_ if t.topic_id != -1]
rate_topics(keywords, client)                               # 0-3 per topic
word_intrusion(keywords, client).intruder_probability       # automated word-intrusion test
labels, conf = assign_documents(model, new_docs, client, allow_other=True)   # -1 = none fits
model.reduce_outliers(strategy="decisions", decisions_client=client)
apply_merges(model, suggest_merges(model, client, threshold=0.5))
```

Merge suggestions are conservative and often empty at 0.5; lower the threshold to see borderline pairs and
check them before merging.

## 10. Start from a codebook (seeded topics)

Script: [`examples/09_seeded_topics.py`](../examples/09_seeded_topics.py)

When you already know (some of) the categories, describe each in one sentence or a list of words:

```python
seeds = {
    "Space": "Space flight: NASA, rockets, satellites, orbits, the moon and planets",
    "Cars": "Cars: engines, driving, car models, dealers, tires and fuel",
    "Medicine": ["disease", "doctor", "patient", "treatment", "drug"],     # word lists work too
}
model = TriTopic().fit(documents, seeds=seeds)

model.get_topic_info()[["Topic", "Seed", "Size", "Keywords"]]
model.seed_topics_          # {seed name: topic id}; TopicInfo.seed holds the seed name
model.emergent_topics_      # ids of the topics nobody asked for: the interesting part
model.seed_anchors_         # the documents pinned to each seed
```

The best-matching documents of each seed are pinned to one topic; all other documents join a seeded topic
or form new ones. A seed that fits no document is ignored with a warning, which shows codebook categories
missing from the data.

- With pre-computed embeddings TriTopic encodes the seeds with `embedding_model`. If your embeddings come
  from another model, pass `seed_embeddings=encoder.encode(list(seeds.values()))`.
- `TriTopicConfig(seed_anchors=20)` sets the number of anchors per seed (default 1% of the corpus, 5-30).
- Seed every theme you know: with a partial codebook a seeded topic can absorb a neighbouring unseeded theme.
  `TopicInterpreter.refine()` splits such mixed topics afterwards.

Effect on benchmark data: full codebook ARI 0.515 → 0.624 (dev), 0.399 → 0.498 (held-out); BBC demo with
five one-line seeds NMI 0.742 → 0.801, ARI 0.631 → 0.800.

## 11. Research toolkit

Script: [`examples/10_research_toolkit.py`](../examples/10_research_toolkit.py)

```python
from tritopic.research import (topic_reliability, saturation_curve, plot_saturation,
                               bridge_documents, topic_connections, topic_prevalence, compare_groups,
                               distinctive_keywords, topic_evolution, plot_evolution,
                               topic_quotes, methods_report)
```

**Which topics can I trust?**

```python
rel = topic_reliability(model)            # refits on 5 random 80% samples
rel[["topic", "label", "reliability", "core_share", "reliable"]]
model.document_stability_                 # per document: share of refits it stayed with its topic
```

Report topics with reliability below 0.7 with care, or merge them. `method="consensus"` is instant
(uses the stored Leiden runs) but a weaker signal.

**Do I have enough data?**

```python
sat = saturation_curve(model)
print(sat)                                # e.g. "all topics discoverable with 30% of the documents"
sat.summary                               # fraction, n_docs, recovered
sat.point_95, sat.point_all, sat.saturated, sat.late_topics
plot_saturation(sat).show()
```

If `late_topics` is not empty, those topics only appear with (nearly) all documents: collect more data
before drawing conclusions about them.

**Where do topics meet?**

```python
bridge_documents(model, top_n=10)         # doc, topic, other_topic, bridge_score, text
topic_connections(model).head()           # topic_a, topic_b, strength, bridge_docs
```

**How big is each topic, and does it differ between groups?**

```python
topic_prevalence(model)                                   # share with 95% Wilson interval
topic_prevalence(model, groups=df["country"])             # per group
compare_groups(model, df["country"])                      # chi2 / Fisher, Cramér's V, BH-adjusted p
distinctive_keywords(model, df["country"], "AT", "DE", n=10)              # words that separate groups
distinctive_keywords(model, df["country"], "AT", "DE", topic_id=3)        # ... within one topic
```

**How do topics develop over time?**

```python
evo = topic_evolution(model, df["date"], freq="Q")        # "M", "Q", "Y"
evo.events                                                # birth / continuation / split / merge / death
evo.lineage(3)                                            # period topics of global topic 3
plot_evolution(evo).show()                                # Sankey diagram
```

**Quotes and the methods section**

```python
topic_quotes(model, n=3)                                  # citable sentences per topic
print(methods_report(model, corpus="12,400 hotel reviews from 2023-2025"))
```

`methods_report` is written from the fitted model without an LLM, so it is reproducible. It includes the
parameter table and software versions; check and adapt the wording before publishing.

## 12. Codebook and second coder for manual coding

Script: [`examples/11_codebook_and_second_coder.py`](../examples/11_codebook_and_second_coder.py)

```python
from tritopic import TopicInterpreter
from tritopic.integrations.decisions import DecisionsClient, intercoder_reliability

interp = TopicInterpreter(domain_hint="interview transcripts")
interp.interpret(model)                                   # labels + descriptions first

codebook = interp.codebook(model, n_quotes=3)             # name, definition, inclusion, exclusion,
codebook.to_excel("codebook.xlsx")                        # coding_notes, anchor_examples (real quotes)

ic = intercoder_reliability(model, DecisionsClient(), sample_size=200)
print(ic.kappa, ic.agreement)                             # Cohen's kappa, raw agreement
ic.per_topic                                              # precision / recall / F1 per topic
ic.confusion                                              # rows: TriTopic, columns: LLM
```

The second coder sees only what a human coder would see in a codebook (label, description, keywords), so
kappa measures whether the topics can be applied consistently. Kappa above 0.8 counts as almost perfect
agreement, 0.6-0.8 as substantial (Landis & Koch, 1977). Low F1 for a topic points to a category that is
hard to apply: sharpen its definition or merge it.

## 13. Evaluation and visualization

Script: [`examples/08_evaluate_and_visualize.py`](../examples/08_evaluate_and_visualize.py)

```python
model.evaluate()         # coherence (NPMI, whole corpus), diversity, stability, outlier ratio

from sklearn.metrics import normalized_mutual_info_score
normalized_mutual_info_score(true_labels, model.labels_)        # if you have reference labels

model.visualize()                       # 2D document map
model.visualize_topics(n_keywords=8)    # keyword bars
model.visualize_hierarchy()             # dendrogram
model.visualize_overlap(threshold=0.1)  # topic co-occurrence
model.visualize_hierarchy_tree()        # after build_hierarchy()
```

All visualizations return Plotly figures (`.show()`, `.write_html()`).

## 14. Configuration recipes

```python
from tritopic import TriTopicConfig

# Short texts (tweets, reviews): denser graph, smaller clusters allowed
TriTopicConfig(n_neighbors=20, min_cluster_size=5)

# Long, technical documents: more weight on wording
TriTopicConfig(semantic_weight=0.4, lexical_weight=0.4)

# Large corpora (50k+): fewer consensus runs, fewer refinement iterations
TriTopicConfig(n_consensus_runs=5, max_iterations=3)

# Broader topics without fixing the number
TriTopicConfig(auto_resolution_tolerance=0.10)

# Reproducibility
TriTopicConfig(random_state=42)
```

All parameters with defaults: [technical documentation §15](docs.md#15-configuration-reference).

## 15. FAQ and troubleshooting

**The first fit is slow.** The embedding model is downloaded and the documents are encoded once; with
pre-computed embeddings a fit of 2,000 documents takes about 7-10 s. UMAP compiles with numba on first use.

**Too many / too few topics.** Pass `n_topics=k`, use `reduce_topics(k)`, or raise/lower
`auto_resolution_tolerance`. With known categories, use seeds.

**One topic mixes two themes.** `model.divide(topic_id, n_subtopics=2)`, or let
`TopicInterpreter.refine()` decide.

**Results differ between runs.** Set `random_state`. Consensus clustering keeps runs close (NMI spread
0.014 across seeds in the benchmark); `topic_reliability()` shows which topics are unstable.

**Keywords contain domain boilerplate** (e.g. "hotel" in hotel reviews). Words in nearly every document get
a low IDF weight already; to remove them completely, delete them from the texts before fitting (single
frequent words hardly change the embeddings).

**No OpenAI key.** All non-LLM features work without a key. `SimpleLabeler` gives keyword labels.

**Dropbox / OneDrive folders.** Keep virtual environments and model caches outside synced folders.
