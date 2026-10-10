"""
05 - Metadata, topics over time, and other languages.

Run:  python examples/05_metadata_time_and_languages.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from tritopic import TopicVisualizer, TriTopic, TriTopicConfig

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.sport.hockey", "talk.politics.mideast", "sci.crypt"])
keep = [i for i, d in enumerate(data.data) if len(d.strip()) > 50]
documents = [data.data[i] for i in keep]

# Example metadata. In real projects: source, author, product, rating, date, ...
rng = np.random.default_rng(0)
metadata = pd.DataFrame({
    "source": rng.choice(["forum", "mailing list", "news"], len(documents)),       # categorical
    "rating": rng.integers(1, 6, len(documents)),                                   # numerical
    "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, len(documents)), unit="D"),
})

# --- 1. Metadata view ------------------------------------------------------------------
# Metadata strengthens edges between documents that are already similar in content and
# share metadata; it never connects unrelated documents.
config = TriTopicConfig(use_metadata_view=True, metadata_weight=0.2, verbose=False)
model = TriTopic(config=config, n_topics=4)
model.fit(documents, metadata=metadata)
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))

# Topic share per metadata value
table = pd.crosstab(metadata["source"], model.labels_, normalize="index").round(2)
print("\nTopic share per source:\n", table)

# --- 2. Topics over time ----------------------------------------------------------------
fig = TopicVisualizer().plot_topic_over_time(labels=model.labels_, timestamps=metadata["date"].tolist(),
                                             topics=model.topics_)
fig.write_html("topics_over_time.html")
monthly = pd.crosstab(metadata["date"].dt.to_period("M"), model.labels_)
print("\nDocuments per topic and month:\n", monthly.head())

# --- 3. Other languages -----------------------------------------------------------------
# Stop words for keywords:  "english", "german", "french", "spanish", "multilingual"
german = TriTopic(language="german")                    # German stop words, default embedding model
mixed = TriTopic(language="multilingual")               # switches to BAAI/bge-m3 (100+ languages)
custom = TriTopic(language="german", embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
print("\nlanguage settings:", german.config.language, mixed.config.embedding_model, custom.config.embedding_model)
