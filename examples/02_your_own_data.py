"""
02 - Your own data: CSV in, configured model, results back into the table.

Run:  python examples/02_your_own_data.py
The script first writes a small demo CSV so it runs anywhere; replace it with your own file.
"""
import re

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic, TriTopicConfig

# --- demo CSV (replace with your own) ------------------------------------------------------
news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["rec.autos", "talk.politics.guns", "sci.electronics"])
pd.DataFrame({"id": range(len(news.data)), "text": news.data}).to_csv("my_texts.csv", index=False)

# --- 1. Load and clean ------------------------------------------------------------------
df = pd.read_csv("my_texts.csv")
df["text"] = (df["text"].fillna("")
              .map(lambda t: re.sub(r"\s+", " ", t).strip()))       # collapse whitespace
df = df[df["text"].str.split().str.len() >= 10].reset_index(drop=True)  # drop very short texts
documents = df["text"].tolist()
print(f"{len(documents)} documents")

# --- 2. Optional: compute embeddings yourself (any sentence-transformers model) -----------
# Useful to try several TriTopic settings without re-encoding, or to use a GPU / other model.
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")      # e.g. "all-mpnet-base-v2", "BAAI/bge-m3"
embeddings = encoder.encode(documents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

# --- 3. Configure ---------------------------------------------------------------------
config = TriTopicConfig(
    n_neighbors=15,            # larger -> smoother, broader topics
    min_cluster_size=10,       # smaller clusters become outliers (-1)
    n_keywords=10,
    keyword_method="ctfidf",   # or "bm25", "keybert"
    random_state=42,
    verbose=False,
)
model = TriTopic(config=config, n_topics=8)            # exactly 8 topics
labels = model.fit_transform(documents, embeddings=embeddings)

# --- 4. Write results back -------------------------------------------------------------
info = model.get_topic_info().set_index("Topic")
df["topic"] = labels
df["topic_keywords"] = df["topic"].map(info["Keywords"])
df["topic_probability"] = model.probabilities_.max(axis=1)   # confidence of the assignment
df.to_csv("my_texts_with_topics.csv", index=False)
print(df[["topic", "topic_keywords", "topic_probability"]].head())
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))
