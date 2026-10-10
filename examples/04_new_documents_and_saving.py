"""
04 - Assign new documents and save/load a model.

Run:  python examples/04_new_documents_and_saving.py
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic

cats = ["sci.space", "rec.autos", "sci.med", "comp.graphics"]
train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats)
documents = [d for d in train.data if len(d.strip()) > 50]

model = TriTopic(n_topics=4, verbose=False).fit(documents)
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))

# --- 1. Hard labels for new documents ----------------------------------------------------
new_docs = [
    "NASA plans a new mission to put a rover on the surface of Mars.",
    "My car's engine makes a strange noise when I brake on the highway.",
    "The patient was treated with antibiotics after the infection got worse.",
    "Which file format is best for storing 3D models and textures?",
]
print("\nAssigned topics:", model.transform(new_docs))      # -1 if below model.outlier_threshold_ (calibrated on the training data)

# --- 2. Topic probabilities --------------------------------------------------------------
proba = model.transform_proba(new_docs)                      # shape (n_new_docs, n_topics)
topic_ids = [t.topic_id for t in model.topics_ if t.topic_id != -1]
for doc, p in zip(new_docs, proba):
    best = int(np.argmax(p))
    print(f"{p[best]:.2f}  topic {topic_ids[best]}  <- {doc[:60]}")

# --- 3. Probabilities for the training documents ------------------------------------------
print("\nprobabilities_ shape:", model.probabilities_.shape)  # columns follow topic_ids

# --- 4. Save and load (embeddings, reducer, topics, labels, LLM labels) --------------------
model.save("tritopic_model.pkl")
loaded = TriTopic.load("tritopic_model.pkl")
print("Same predictions after loading:", np.array_equal(loaded.transform(new_docs), model.transform(new_docs)))
