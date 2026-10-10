"""
10 - Research toolkit: the numbers you need for a paper.

Run:  python examples/10_research_toolkit.py

reliability per topic, saturation, bridge documents, prevalence with confidence
intervals, group comparison with tests, topic evolution, quotes and a methods paragraph.
None of these steps needs an LLM.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic, TriTopicConfig
from tritopic.research import (bridge_documents, compare_groups, distinctive_keywords, methods_report,
                               saturation_curve, topic_connections, topic_evolution, topic_prevalence,
                               topic_quotes, topic_reliability)

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.autos", "sci.med", "comp.graphics"])
keep = [i for i, d in enumerate(data.data) if len(d.strip()) > 50]
documents = [data.data[i] for i in keep]
newsgroup = np.array(data.target_names)[data.target[keep]]

model = TriTopic(config=TriTopicConfig(verbose=False, random_state=42)).fit(documents)
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))

# --- 1. Reliability: which topics survive refitting on 80% samples? -------------------------
rel = topic_reliability(model)                  # 5 refits; method="consensus" is instant but weaker
print("\nReliability (>= 0.7 = reliable):\n", rel.round(2).to_string(index=False))
# per-document stability is now in model.document_stability_

# --- 2. Saturation: would more data reveal new topics? ---------------------------------------
sat = saturation_curve(model)                   # exact accumulation curve, no refitting
print("\n", sat)
print(sat.summary[["fraction", "n_docs", "recovered"]].round(2).to_string(index=False))
# plot_saturation(sat).show()   (plotly)

# --- 3. Bridges: documents between two topics -------------------------------------------------
br = bridge_documents(model, top_n=5)
for _, b in br.iterrows():
    print(f"\nTopic {b.topic} <-> {b.other_topic} ({b.bridge_score:.0%} of neighbours): {b.text[:120]}...")
print("\nStrongest topic connections:\n", topic_connections(model).head(5).round(3).to_string(index=False))

# --- 4. Prevalence with 95% confidence intervals ---------------------------------------------
print("\n", topic_prevalence(model).round(3).to_string(index=False))           # Wilson intervals
# per group: topic_prevalence(model, groups=newsgroup)

# --- 5. Group comparison: chi-square / Fisher, Cramér's V, Benjamini-Hochberg ----------------
length = np.where([len(d.split()) > 100 for d in documents], "long", "short")
print("\n", compare_groups(model, length).round(3).to_string(index=False))
print("\nWords typical for long vs. short posts:\n",
      distinctive_keywords(model, length, "long", "short", n=6).round(2).to_string(index=False))

# --- 6. Topic evolution: births, splits, merges and deaths over time --------------------------
# 20 Newsgroups has no usable dates, so we simulate: graphics posts only appear in 2024.
# Real dated news: examples/12_topic_evolution_news.py
rng = np.random.default_rng(0)
dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, len(documents)), unit="D")
late = newsgroup == "comp.graphics"
dates = dates.where(~late, pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, late.sum()), unit="D"))
evo = topic_evolution(model, dates, freq="Y")
print("\nEvents:\n", evo.events.to_string(index=False))
# plot_evolution(evo).show()    (Sankey diagram)

# --- 7. Quotes: citable sentences per topic ---------------------------------------------------
for _, q in topic_quotes(model, n=1).iterrows():
    print(f"\nTopic {q.topic}: \"{q.quote}\"")

# --- 8. Methods paragraph for the paper --------------------------------------------------------
print("\n" + methods_report(model, corpus="four newsgroups of the 20 Newsgroups corpus"))
