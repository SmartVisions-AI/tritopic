"""
12 - Topic evolution on real dated news: which themes are born, split, merge and die?

Run:  python examples/12_topic_evolution_news.py
Data: HuffPost News Category dataset (Misra, 2022), 2012-2022, downloaded from the Hugging Face hub
      (about 90 MB, once).  We sample 700 articles per year.

On this corpus topic_evolution() finds the COVID pandemic as a new topic in 2020 (continuing in 2021
and 2022) and the war in Ukraine in 2022, together with smaller events such as the Flint water
crisis (2016), the Hong Kong protests (2019) and the Roe v. Wade decision (2022).
"""
import numpy as np
import pandas as pd
from datasets import load_dataset

from tritopic import TriTopic, TriTopicConfig
from tritopic.research import plot_evolution, topic_evolution

df = load_dataset("heegyu/news-category-dataset", split="train").to_pandas()
df["text"] = (df.headline.fillna("") + ". " + df.short_description.fillna("")).str.strip()
df = df[df.text.str.split().str.len() >= 8]
df["year"] = pd.to_datetime(df.date).dt.year
df = df.loc[np.concatenate([g.sample(min(len(g), 700), random_state=0).index for _, g in df.groupby("year")])]
df = df.sort_values("date").reset_index(drop=True)

model = TriTopic(config=TriTopicConfig(verbose=False, random_state=42)).fit(df.text.tolist())
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))

# --- 1. Topics per year, linked across years ---------------------------------------------------
# Each year is clustered on its own (finer than the global model: resolution = 4 x model.resolution_)
# and topics of consecutive years are linked when their centroids have cosine >= 0.75.
evo = topic_evolution(model, df.date, freq="Y")
print("\n", evo.events.event.value_counts().to_string())

# --- 2. What was new? ----------------------------------------------------------------------------
births = evo.events[evo.events.event == "birth"]
print("\nNew topics per year:")
for r in births.itertuples():
    print(f"  {r.period}  {r.size:4d} articles  {r.keywords}")

# --- 3. Follow one theme -------------------------------------------------------------------------
covid = evo.nodes[evo.nodes.keywords.str.contains("covid|coronavirus")]
print("\nCOVID topics:\n", covid[["node", "period", "size", "keywords"]].to_string(index=False))
print("Successors of the first COVID topic:\n",
      evo.links[evo.links.source == covid.node.iloc[0]].round(3).to_string(index=False))

# The documents of a period topic: evo.nodes.docs  (indices into df)
first = covid.iloc[0]
print("\nExample headlines:", df.headline.iloc[first.docs[:3]].tolist())

plot_evolution(evo).write_html("topic_evolution.html")     # Sankey diagram
print("Saved topic_evolution.html")
