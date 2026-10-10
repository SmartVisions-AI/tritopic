# TriTopic

**Graph-based topic modeling that fuses meaning, wording and metadata, and lets an LLM read the result.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-133%20passed-brightgreen.svg)](tests/)

TriTopic builds one graph over your documents from three views (sentence embeddings, TF-IDF wording and
optional metadata), finds topics with consensus Leiden clustering, and describes them with keywords that are
both frequent and spread across the topic. An optional LLM step labels each topic, explains it, and splits
topics that turn out to mix unrelated themes.

**New in 2.5: built for research.** Start from the categories you expect and let the rest emerge
([codebook mode](#start-from-a-codebook-seeded-topics)). Then get what a paper needs and no other topic
model gives you ([research toolkit](#research-toolkit)): reliability per topic, a saturation curve
("have we seen every topic?"), bridge documents, prevalence with confidence intervals, group comparisons
with significance tests, topic births and splits over time, citable quotes, a coding scheme for manual
coding, an LLM second coder with Cohen's kappa, and a ready-made methods paragraph.

In a head-to-head benchmark with **identical embeddings**, TriTopic 2.4 beats BERTopic on every quality measure:

| Fixed topic count, 4 datasets × 5 k × 3 seeds | TriTopic 2.4 | BERTopic (tuned) | BERTopic (default) |
|---|---|---|---|
| NMI against the class labels | **0.582** | 0.439 | 0.439 |
| ARI | **0.440** | 0.285 | 0.285 |
| Keyword coherence (NPMI, whole corpus) | **0.262** | 0.117 | 0.048 |
| Documents left without a topic | **0%** | 18.4% | 18.4% |
| NMI spread across seeds | **0.014** | 0.105 | 0.105 |
| LLM rating of the topics (0-3) | **2.21** | 1.94 | 0.49 |

TriTopic has the better NMI in 18 of 20 dataset/k settings. Methodology and the cases where BERTopic is
ahead are in [Benchmarks](#benchmarks).

**See it on real data:** [interactive demo on 1,000 BBC articles](https://htmlpreview.github.io/?https://github.com/SmartVisions-AI/tritopic/blob/main/examples/bbc_demo/index.html)
(source: [`examples/bbc_demo/index.html`](examples/bbc_demo/index.html)): six runs against BERTopic, LLM
interpretation, codebook seeds and every research output.

**Learn it:** [user guide with code for every task](docs/user_guide.md) · [11 runnable examples](examples/) ·
[technical documentation](docs/docs.md)

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples and user guide](#examples-and-user-guide)
- [Let an LLM interpret the topics](#let-an-llm-interpret-the-topics)
- [Start from a codebook (seeded topics)](#start-from-a-codebook-seeded-topics)
- [Research toolkit](#research-toolkit)
- [How it works](#how-it-works)
- [Choosing the number of topics](#choosing-the-number-of-topics)
- [Working with a fitted model](#working-with-a-fitted-model)
- [LLM judgements with the OpenAI Decisions API](#llm-judgements-with-the-openai-decisions-api)
- [Configuration](#configuration)
- [Evaluation metrics](#evaluation-metrics)
- [Benchmarks](#benchmarks)
- [Citation](#citation)

---

## Installation

```bash
pip install git+https://github.com/SmartVisions-AI/tritopic.git        # 2.5 (this repository)
pip install tritopic                                                    # latest PyPI release (2.3.0)
```

Extras: `[llm]` (OpenAI/Anthropic SDKs for `LLMLabeler`), `[turftopic]`, `[benchmark]` (BERTopic,
datasets), `[full]`, `[dev]`. Python 3.9-3.13.

The LLM features `TopicInterpreter` and the Decisions API integration need only an `OPENAI_API_KEY`
environment variable; they call the API over HTTPS without extra packages.

## Quick start

```python
from tritopic import TriTopic

model = TriTopic()                    # n_topics="auto": the topic count adapts to the corpus
labels = model.fit_transform(documents)

model.get_topic_info()                # topic id, size, keywords, label, coherence
model.get_representative_docs(0, n_docs=3)
model.visualize()                     # interactive 2D map (Plotly)
```

```
[TriTopic] Fitting model on 1000 documents
   > Generating embeddings (all-MiniLM-L6-v2)...
   > Reducing dimensions to 10d (umap)...
   > Building lexical similarity matrix...
   > Auto resolution: 0.019 (~6 topics, scanned 15 resolutions by keyword coherence)
   > Starting iterative refinement (max 5 iterations)...
   > Extracting keywords and representative documents...
```

Pre-computed embeddings, a fixed number of topics, another language:

```python
model = TriTopic(n_topics=10, language="german", embedding_model="BAAI/bge-m3")
model.fit(documents, embeddings=my_embeddings)
```

## Examples and user guide

The [user guide](docs/user_guide.md) walks through every task with code. Each step has a runnable script:

| Script | What it shows |
|---|---|
| [`01_quickstart.py`](examples/01_quickstart.py) | Fit, inspect topics, visualize |
| [`02_your_own_data.py`](examples/02_your_own_data.py) | CSV in, cleaning, own embeddings, results back to the table |
| [`03_shape_the_topics.py`](examples/03_shape_the_topics.py) | Topic count, outliers, merge, split, hierarchy |
| [`04_new_documents_and_saving.py`](examples/04_new_documents_and_saving.py) | `transform()`, probabilities, save/load |
| [`05_metadata_time_and_languages.py`](examples/05_metadata_time_and_languages.py) | Metadata view, topics over time, other languages |
| [`06_llm_interpretation.py`](examples/06_llm_interpretation.py) | `TopicInterpreter`: labels, mixed-topic detection, `refine()` |
| [`07_decisions_api.py`](examples/07_decisions_api.py) | LLM ratings, word intrusion, document assignment, merges |
| [`08_evaluate_and_visualize.py`](examples/08_evaluate_and_visualize.py) | Metrics, NMI/ARI, all plots |
| [`09_seeded_topics.py`](examples/09_seeded_topics.py) | Codebook mode: seeded and emergent topics |
| [`10_research_toolkit.py`](examples/10_research_toolkit.py) | Reliability, saturation, bridges, prevalence, group tests, evolution, quotes, methods text |
| [`11_codebook_and_second_coder.py`](examples/11_codebook_and_second_coder.py) | Coding scheme with anchor quotes, LLM second coder (kappa) |

## Let an LLM interpret the topics

`TopicInterpreter` lets an LLM read every topic the way an analyst would. For each topic it gets the ranked
keywords, example documents that cover the topic in proportion (one typical document per region of the
topic, so a mixture becomes visible), and the neighbouring topics for contrast. It returns a label, a
description, the main aspects, and a verdict: one **coherent** theme, a **mix** of distinct themes (with the
sub-themes and the documents behind them), or **unclear**.

```python
from tritopic import TopicInterpreter

interpreter = TopicInterpreter(domain_hint="BBC news articles")   # OPENAI_API_KEY; default model gpt-6-luna
results = interpreter.interpret(model)          # also writes labels/descriptions into model.topics_
for r in results.values():
    print(r.label, "|", r.verdict, "|", r.description)

log = interpreter.refine(model)                 # split topics the LLM judges as mixed
print(interpreter.summarize(model))             # short overview of the whole topic landscape
```

`refine()` splits a topic only when the LLM calls it mixed, at least two sub-themes are each backed by two
or more example documents (one stray article does not make a topic mixed), and the graph split actually
separates the documents the LLM assigned to the different sub-themes. Otherwise the split is undone.

In the BBC demo the LLM notices that one topic mixes UK politics with business news and another mixes
football with rugby. `refine()` separates both (agreement with the BBC sections: NMI 0.742 → 0.781,
ARI 0.631 → 0.739), and the new topics get labels like "Blair-Era UK Politics" and "Business and Economic
News". The default model was chosen on development data:

| Model (dev splits, deliberately coarse topics) | Mixed topics found | refine(): NMI | Time per corpus |
|---|---|---|---|
| **gpt-6-luna** (default) | 9 of 15 | 0.614 → 0.654 | 29 s |
| gpt-5.5 | 8 of 15 | 0.614 → 0.638 | 41 s |
| gpt-5.4-mini | 4 of 15 | 0.614 → 0.625 | 16 s |

On held-out data, `refine()` raised NMI from 0.606 to 0.639; no dataset lost more than 0.001. An
interpretation uses about 1,500 input tokens per topic. `LLMLabeler` (label and description only, Anthropic
or OpenAI) remains available via `model.generate_labels(LLMLabeler(...))`.

## Start from a codebook (seeded topics)

Most topic models make you choose between deductive coding (you define the categories) and inductive
discovery (the algorithm does). TriTopic does both in one fit: describe the topics you expect, and it pins
the documents that match each description best to one topic. All other documents can join a seeded topic
or form **new topics nobody asked for**.

```python
seeds = {
    "Technology": "Technology news: computers, mobile phones, the internet, software and gadgets",
    "Sport": "Sport news: football, rugby, tennis, athletics, matches and players",
    "Politics": ["government", "election", "minister", "parliament"],      # word lists work too
}
model = TriTopic().fit(documents, seeds=seeds)

model.get_topic_info()[["Topic", "Seed", "Size", "Keywords"]]
model.seed_topics_        # {"Technology": 0, "Sport": 1, "Politics": 2}
model.emergent_topics_    # ids of the topics that emerged on their own
```

Seeds are fixed memberships inside every Leiden run, so the graph still decides where all other documents
go. Results: full codebook on the dev splits ARI 0.515 → 0.624, held-out 0.399 → 0.498; in the BBC demo
five one-line seeds raise NMI from 0.742 to 0.801 and ARI from 0.631 to 0.800. A seed that fits no document
is reported, so you also learn which codebook categories are missing from the data. Seed every theme you
know: with a partial codebook a seeded topic can absorb a neighbouring unseeded theme.

## Research toolkit

`tritopic.research` answers the questions reviewers ask about a topic model. Nothing here needs an LLM
unless marked.

```python
from tritopic.research import (topic_reliability, saturation_curve, bridge_documents, topic_connections,
                               topic_prevalence, compare_groups, distinctive_keywords, topic_evolution,
                               topic_quotes, methods_report)

topic_reliability(model)                 # does each topic survive refitting on 80% samples? (0-1)
saturation_curve(model)                  # with how much data does every topic appear?
bridge_documents(model)                  # documents that connect two topics
topic_prevalence(model)                  # share per topic with 95% confidence interval
compare_groups(model, df["country"])     # chi2 / Fisher per topic, Cramér's V, BH-corrected p
distinctive_keywords(model, df["country"], "AT", "DE")   # words that separate two groups
topic_evolution(model, df["date"], freq="Q").events       # births, splits, merges, deaths over time
topic_quotes(model, n=3)                 # citable sentences per topic
print(methods_report(model, corpus="12,400 hotel reviews"))   # methods paragraph + parameter table

interpreter.codebook(model)              # LLM: coding scheme with definitions and real anchor quotes
intercoder_reliability(model, client)    # LLM: second coder, Cohen's kappa and F1 per topic
```

| Output | What makes it new | Validation |
|---|---|---|
| Topic reliability | Per-topic stability under resampling instead of one global score | Topics ≥ 0.7 were 77% pure, < 0.5 only 54% (4 corpora) |
| Saturation curve | The qualitative-research question "have we seen every theme?" answered exactly (topic accumulation curve) | 95% of topics visible after 25-35% of the data, all after 45-65% |
| Bridge documents | Hybrid cases between topics, from the graph | — |
| Group comparison | Significance tests and effect sizes per topic, corrected for multiple testing | — |
| Topic evolution | How topic *content* splits and merges over time, not just frequency | — |
| Codebook + second coder | From topics to a coding scheme for manual content analysis, with inter-coder agreement | BBC demo: kappa 0.94 |
| Methods paragraph | Reproducible description of the analysis for the paper | — |

Details: [user guide §11-12](docs/user_guide.md#11-research-toolkit), [technical documentation §19](docs/docs.md#19-research-toolkit).

## How it works

```
documents ─► embeddings ─► UMAP (10d) ─► semantic kNN graph ─┐
          └► token counts ─► TF-IDF ───► lexical kNN graph ──┼─► fused graph ─► consensus Leiden ─► topics
metadata (optional) ────────────────────► reweights edges ───┘         ▲
                                                     auto resolution (keyword coherence)
topics ─► coverage-weighted c-TF-IDF keywords ─► centroids, probabilities ─► optional LLM interpretation
```

1. **Semantic view.** Sentence-transformer embeddings, reduced to 10 dimensions with UMAP. The kNN graph
   uses Euclidean distance with a self-tuning Gaussian kernel (UMAP output is a Euclidean layout) and mixes
   mutual-kNN and shared-nearest-neighbour edges.
2. **Lexical view.** TF-IDF over the same token counts as the keywords (the corpus is tokenized once),
   connected as a mutual-kNN graph. Edges present in both views get a small bonus. Without this view NMI
   drops by about 0.05.
3. **Metadata view (optional).** Categorical and numerical columns reweight existing edges between documents
   with similar metadata; they never add edges.
4. **Consensus Leiden.** Leiden runs 10 times; for every edge, the share of runs that keep both ends together
   becomes its weight in a consensus graph, which is clustered again until the runs agree (Lancichinetti &
   Fortunato, 2012). Memory grows with the number of edges, not with n².
5. **Iterative refinement.** Reduced embeddings are pulled slightly towards their topic centroid and the
   semantic graph is re-clustered until consecutive partitions agree (ARI ≥ 0.95).
6. **Keywords.** `score(t) = IDF(t) · √(tf_share(t) · coverage(t))`. Term frequency alone favours words
   repeated in a few long documents, document coverage alone favours boilerplate. Tokens must start with two
   letters, so numbers never become keywords.

## Choosing the number of topics

- **`n_topics="auto"`** (default) scans 15 Leiden resolutions, extracts keywords for each candidate
  partition, and keeps the coarsest one whose keyword coherence (NPMI) is within 5% of the best. Partitions
  where one topic holds more than half of the documents are skipped. The scan is in
  `model.resolution_search_`, the choice in `model.resolution_`.
- **`n_topics=k`** searches the resolution for exactly *k* topics (widening the search range if needed) and
  merges down if it overshoots.
- **Fixed resolution:** `TriTopicConfig(auto_resolution=False, resolution=0.3)`.

Auto mode tends to find more topics than a corpus has classes (AG News: about 27 for 4 classes; BBC: about
14 for 5). If you know the granularity you need, pass `n_topics`, or let `TopicInterpreter.refine()` split
a coarse model where needed.

## Working with a fitted model

```python
model.get_topic_info()                        # DataFrame overview
model.get_topic(3).keywords                   # TopicInfo: keywords, scores, size, label, representative docs
model.probabilities_                          # soft assignments (n_docs × n_topics)
model.get_document_topics(doc_idx=0, top_n=3)

model.transform(new_documents)                # assign new documents (nearest centroid; threshold calibrated on the training data)
model.transform_proba(new_documents)

model.reduce_outliers(strategy="neighbors")   # or "embeddings", or "decisions" (LLM, see below)
model.reduce_topics(8)                        # merge the most similar topics until 8 remain
model.merge_topics([2, 7])
model.divide(topic_id=0, n_subtopics=3)       # split one topic

hierarchy = model.build_hierarchy(n_levels=3) # coarse → fine topic tree
model.topic_overlap_matrix(threshold=0.1)

model.evaluate()                              # coherence, diversity, stability, outlier ratio
model.save("model.pkl"); model = TriTopic.load("model.pkl")
```

Visualizations (Plotly): `visualize()`, `visualize_topics()`, `visualize_hierarchy()`,
`visualize_hierarchy_tree()`, `visualize_overlap()`, and `TopicVisualizer.plot_topic_over_time(...)`.

A [turftopic](https://github.com/x-tabdeveloping/turftopic) adapter is available as
`tritopic.integrations.turftopic.TriTopicModel` (`pip install "tritopic[turftopic]"`).

## LLM judgements with the OpenAI Decisions API

`tritopic.integrations.decisions` uses OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(typed answers: probability, choice, score). Nothing runs inside `fit()`.

```python
from tritopic.integrations.decisions import (
    DecisionsClient, word_intrusion, rate_topics, assign_documents, suggest_merges, apply_merges,
    intercoder_reliability)

client = DecisionsClient()                              # OPENAI_API_KEY; model gpt-6-luna
keywords = [t.keywords for t in model.topics_ if t.topic_id != -1]
word_intrusion(keywords, client).intruder_probability   # automated word-intrusion test
rate_topics(keywords, client)                           # 0-3 per topic
assign_documents(model, new_documents, client)          # LLM assignment to topics
model.reduce_outliers(strategy="decisions", decisions_client=client)
apply_merges(model, suggest_merges(model, client))      # merge pairs the LLM considers the same theme
intercoder_reliability(model, client).kappa             # LLM as second coder (Cohen's kappa)
```

On the benchmark data the LLM rates TriTopic's topics 2.21 vs. 1.94 for tuned BERTopic. LLM assignment of
held-out documents is slightly better than the centroid rule (0.657 vs. 0.648); with an "other" option it
abstains on unclear documents and is about 74% accurate on the rest. Merge suggestions are safe but
conservative. The full evaluation (about 3,500 requests) cost around USD 0.15.

## Configuration

```python
from tritopic import TriTopic, TriTopicConfig

config = TriTopicConfig(
    language="english",                 # stop words: english, german, french, spanish, multilingual
    embedding_model="all-MiniLM-L6-v2",
    use_dim_reduction=True, reduced_dims=10, reduced_metric="euclidean",
    n_neighbors=15, graph_type="hybrid", snn_weight=0.5,
    use_lexical_view=True, use_metadata_view=False,
    semantic_weight=0.5, lexical_weight=0.3, metadata_weight=0.2,
    auto_resolution=True, auto_resolution_tolerance=0.05, auto_resolution_max_share=0.5,
    resolution=0.3,                     # start of the n_topics=k search; used if auto_resolution=False
    n_consensus_runs=10, min_cluster_size=5,
    use_iterative_refinement=True, max_iterations=5, convergence_threshold=0.95,
    keyword_method="ctfidf",            # or "bm25", "keybert"
    n_keywords=10, soft_assignment_method="centroid",
    random_state=42,
)
model = TriTopic(config=config)
```

## Evaluation metrics

`model.evaluate()` returns mean and standard deviation of NPMI coherence (computed over the whole corpus;
word pairs that never co-occur count as -1), keyword diversity, consensus stability (mean pairwise ARI of
the Leiden runs) and the outlier ratio. Standalone functions live in `tritopic.utils.metrics`:
`compute_coherence_batch`, `compute_diversity`, `compute_silhouette`, `compute_downstream_score`.

## Benchmarks

Head-to-head with BERTopic 0.17.4 on the datasets of the original TriTopic benchmark: 20 Newsgroups (2,000
docs), BBC News (1,225), AG News (2,000), arXiv (2,000); five topic counts per dataset plus automatic mode;
seeds 42, 123, 456.

- **Same embeddings for every model:** `all-MiniLM-L6-v2`, pre-computed once per dataset.
- **Two BERTopic variants:** defaults, and "tuned" with BERTopic's keyword best practice
  (`CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)`,
  `ClassTfidfTransformer(reduce_frequent_words=True)`); UMAP seed fixed.
- **No tuning on the evaluation data:** every 2.4 design decision was made on separate development splits
  (20NG test, BBC test, AG News train sample, arXiv validation).
- **Metrics:** NMI/ARI on all documents (BERTopic outliers form one class); strict NPMI over the whole
  corpus; fit time with pre-computed embeddings on one machine.

#### Fixed topic count

| Model | NMI | ARI | NPMI | Diversity | Outliers | Fit time (s) |
|---|---|---|---|---|---|---|
| **TriTopic 2.4** | **0.582** | **0.440** | **0.262** | **0.94** | **0%** | 9.5 |
| TriTopic 2.3 | 0.581 | 0.436 | 0.212 | 0.92 | 0% | 44.3 |
| BERTopic (tuned) | 0.439 | 0.285 | 0.117 | 0.89 | 18.4% | 16.6 |
| BERTopic (default) | 0.439 | 0.285 | 0.048 | 0.44 | 18.4% | 9.9 |

#### Automatic topic count

| Model | NMI | ARI | NPMI | Outliers | Topics found |
|---|---|---|---|---|---|
| **TriTopic 2.4** | **0.560** | **0.398** | **0.299** | **0%** | 21.4 |
| TriTopic 2.3 | 0.528 | 0.302 | 0.200 | 0% | 30.0 |
| BERTopic (tuned) | 0.443 | 0.247 | 0.123 | 18.4% | 26.8 |
| BERTopic (default) | 0.443 | 0.247 | 0.109 | 18.4% | 26.8 |

#### NMI per dataset (fixed k)

| Dataset | TriTopic 2.4 | BERTopic |
|---|---|---|
| 20 Newsgroups | **0.529** | 0.264 |
| AG News | **0.526** | 0.377 |
| arXiv | **0.568** | 0.450 |
| BBC News | **0.702** | 0.663 |

**Where TriTopic does not lead:** on BBC News, BERTopic has the higher ARI for k = 5-20 (NMI is close). On
short texts BERTopic is 1-2 s per run faster; on long documents (arXiv) TriTopic is about twice as fast
(23 s vs. 50 s). BERTopic collapsed to 4 topics on 20 Newsgroups for two of three seeds, which is part of
its lower average there.

Earlier releases quoted mean NMI 0.575 vs. 0.513 for BERTopic. That benchmark gave BERTopic stronger
embeddings (`all-mpnet-base-v2`) than TriTopic and used a lenient coherence measure; the numbers above
supersede it.

**Reproduce:**

```bash
pip install "tritopic[benchmark]"
python benchmarks/compare_bertopic.py --pkg . --models tritopic,bertopic,bertopic_tuned --out results.csv
python benchmarks/summarize.py results.csv
python benchmarks/llm_eval.py --pkg . --out llm.csv     # LLM judgements; needs OPENAI_API_KEY, --dry-run estimates cost
```

The decisions behind 2.4 (keyword scoring, auto resolution, ablations, LLM experiments) are documented in
[`benchmarks/experiments/README.md`](benchmarks/experiments/README.md). Release notes: [CHANGELOG.md](CHANGELOG.md).

## Citation

```bibtex
@software{tritopic2026,
  author = {Egger, Roman},
  title  = {TriTopic: Tri-Modal Graph Topic Modeling},
  year   = {2026},
  url    = {https://github.com/SmartVisions-AI/tritopic}
}
```

## License

MIT, see [LICENSE](LICENSE). Issues and pull requests are welcome.
Maintained by [SmartVisions](https://www.smartvisions.at).
