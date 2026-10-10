"""
01 - Quick start: fit TriTopic and look at the topics.

Run:  python examples/01_quickstart.py
Data: 20 Newsgroups (downloaded by scikit-learn on first use).
"""
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic

# --- 1. Documents: a list of strings -------------------------------------------------
data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.sport.baseball", "sci.med", "comp.graphics"])
documents = [d for d in data.data if len(d.strip()) > 50][:1500]

# --- 2. Fit ----------------------------------------------------------------------------
# n_topics="auto" (default) lets TriTopic pick the number of topics.
# The first run downloads the embedding model (all-MiniLM-L6-v2, ~90 MB).
model = TriTopic(verbose=True)
labels = model.fit_transform(documents)          # one topic id per document, -1 = outlier

# --- 3. Overview -----------------------------------------------------------------------
info = model.get_topic_info()                    # pandas DataFrame
print(info[["Topic", "Size", "Keywords"]].to_string(index=False))

# --- 4. One topic in detail ------------------------------------------------------------
topic = model.get_topic(info.Topic.iloc[0])      # TopicInfo of the largest topic
print("\nKeywords:", topic.keywords)
print("Scores:  ", [round(s, 4) for s in topic.keyword_scores[:5]])
for doc_id, text in model.get_representative_docs(topic.topic_id, n_docs=2):
    print(f"\n[{doc_id}] {text[:200]}...")

# --- 5. One document in detail ---------------------------------------------------------
print("\nDocument 0 belongs to topic", labels[0])
print("Top topics with probabilities:", model.get_document_topics(doc_idx=0, top_n=3))

# --- 6. Interactive map ----------------------------------------------------------------
fig = model.visualize(title="20 Newsgroups - TriTopic")
fig.write_html("topic_map.html")                 # open in a browser
print("\nSaved topic_map.html")
