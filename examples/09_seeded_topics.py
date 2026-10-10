"""
09 - Codebook mode: start from the topics you expect, let the rest emerge.

Run:  python examples/09_seeded_topics.py

Describe each expected topic in one sentence. TriTopic pins the documents that fit a
description best to one topic per seed; all other documents can join a seeded topic or
form new (emergent) topics. Seeds you list but that are not in the data stay small or
empty, so a codebook can be checked against the data.
"""
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic, TriTopicConfig

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.autos", "sci.med", "comp.graphics", "talk.politics.guns"])
documents = [d for d in data.data if len(d.strip()) > 50]

# --- 1. Seeds: topic name -> short description ------------------------------------------
# Only three of the five themes are seeded; graphics and guns have to emerge on their own.
seeds = {
    "Space": "Space flight: NASA, rockets, satellites, orbits, the moon and planets",
    "Cars": "Cars: engines, driving, car models, dealers, tires and fuel",
    "Medicine": "Medicine: diseases, doctors, patients, treatment, drugs and symptoms",
}
model = TriTopic(config=TriTopicConfig(verbose=False, random_state=42))
model.fit(documents, seeds=seeds)

print(model.get_topic_info()[["Topic", "Seed", "Size", "Keywords"]].to_string(index=False))

# --- 2. Seeded vs. emergent topics --------------------------------------------------------
print("\nSeeded topics (seed -> topic id):", model.seed_topics_)
print("Emergent topics:", [(tid, ", ".join(model.get_topic(tid).keywords[:4])) for tid in model.emergent_topics_])

# --- 3. Options ------------------------------------------------------------------------------
# * seed_anchors:        documents pinned per seed (default: 1% of the corpus, 5 to 30)
# * seed_keyword_weight: weight of the seed words vs. the semantic match (default 0.5)
# * seed_embeddings:     pass your own seed vectors when you fit with precomputed embeddings
#   model.fit(docs, embeddings=emb, seeds=seeds, seed_embeddings=my_model.encode(list(seeds.values())))
#
# Seeds still go through the normal pipeline: reduce_topics(), refine() with an LLM,
# transform() and save()/load() keep the seed names.
# Tip: seed every theme you know. With a partial codebook a seeded topic can absorb a
# neighbouring unseeded theme; refine() (examples/06) splits such mixed topics.
