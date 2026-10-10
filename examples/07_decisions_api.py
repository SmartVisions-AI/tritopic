"""
07 - Typed LLM judgements with the OpenAI Decisions API: rate topics, assign documents, merge.

Run:  OPENAI_API_KEY=sk-... python examples/07_decisions_api.py
Cost: well under one cent for this example.
"""
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic
from tritopic.integrations.decisions import (DecisionsClient, apply_merges, assign_documents,
                                             rate_topics, suggest_merges, word_intrusion)

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.autos", "sci.med", "comp.graphics"])
documents = [d for d in data.data if len(d.strip()) > 50]
model = TriTopic(n_topics=8, verbose=False).fit(documents)

client = DecisionsClient()                      # reads OPENAI_API_KEY; model gpt-6-luna
keywords = [t.keywords for t in model.topics_ if t.topic_id != -1]

# --- 1. How interpretable are the topics? -------------------------------------------------
ratings = rate_topics(keywords, client)         # 0 = unrelated ... 3 = one clear theme
for kws, r in zip(keywords, ratings):
    print(f"{r:.2f}  {', '.join(kws[:6])}")
test = word_intrusion(keywords, client)
print(f"Word intrusion: accuracy {test.accuracy:.2f}, probability on the intruder {test.intruder_probability:.2f}")

# --- 2. Assign documents with the LLM ------------------------------------------------------
new_docs = ["The shuttle docked with the space station after a delayed launch.",
            "Which video card renders OpenGL scenes fastest?"]
labels, confidence = assign_documents(model, new_docs, client)            # always picks a topic
print("LLM assignment:", labels, confidence.round(2))
labels, confidence = assign_documents(model, new_docs, client, allow_other=True)  # may answer -1
print("With abstention:", labels)

# --- 3. Outliers via LLM (abstains when no topic fits) --------------------------------------
model.reduce_outliers(strategy="decisions", decisions_client=client)

# --- 4. Merge topics the LLM considers the same theme ----------------------------------------
# The judgement is conservative: with threshold=0.5 it often suggests nothing, which is fine.
# A lower threshold shows the borderline pairs; inspect them before merging.
merges = suggest_merges(model, client, n_pairs=10, threshold=0.5)
print("Merge suggestions at 0.5 (topic a, topic b, probability):", merges)
if not merges:
    print("Borderline pairs at 0.2:", suggest_merges(model, client, n_pairs=10, threshold=0.2))
apply_merges(model, merges)
print(model.get_topic_info()[["Topic", "Size", "Keywords"]].to_string(index=False))
print(f"{client.n_requests} requests")
