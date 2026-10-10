"""
03 - Shape the topics after fitting: granularity, outliers, merging, splitting, hierarchy.

Run:  python examples/03_shape_the_topics.py
"""
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic, TriTopicConfig


def show(model, title):
    info = model.get_topic_info()
    n = len(info[info.Topic != -1])
    print(f"\n== {title}: {n} topics, {int((model.labels_ == -1).sum())} outliers")
    print(info[["Topic", "Size", "Keywords"]].head(8).to_string(index=False))


data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
documents = [d for d in data.data if len(d.strip()) > 50][:3000]

# --- 1. How many topics? ---------------------------------------------------------------
# a) automatic: scans 15 resolutions, keeps the coarsest with near-best keyword coherence
model = TriTopic(verbose=False).fit(documents)
show(model, "auto")
print("chosen resolution:", round(model.resolution_, 3))
print("scan (resolution, topics, coherence):", [(round(r, 3), k, round(c, 3)) for r, k, c in model.resolution_search_][:5], "...")

# b) exact number of topics:   TriTopic(n_topics=20)
# c) fixed Leiden resolution:  TriTopic(config=TriTopicConfig(auto_resolution=False, resolution=0.5))

# --- 2. Outliers -------------------------------------------------------------------------
# Documents in clusters smaller than min_cluster_size get the label -1.
model.reduce_outliers(strategy="neighbors")           # majority vote of the nearest documents
# alternatives: strategy="embeddings" (nearest topic centroid above a threshold),
#               strategy="decisions"  (LLM, see examples/07_decisions_api.py)

# --- 3. Fewer topics: merge the most similar ones ---------------------------------------
model.reduce_topics(10)
show(model, "after reduce_topics(10)")

# --- 4. Merge specific topics -----------------------------------------------------------
ids = model.get_topic_info().Topic.tolist()
model.merge_topics([ids[-1], ids[-2]])                 # the two smallest topics
show(model, "after merge_topics")

# --- 5. Split one topic ------------------------------------------------------------------
largest = model.get_topic_info().Topic.iloc[0]
for sub in model.divide(topic_id=largest, n_subtopics=3):
    print("sub-topic", sub.topic_id, sub.size, sub.keywords[:5])

# --- 6. Topic hierarchy: coarse -> fine ---------------------------------------------------
hierarchy = model.build_hierarchy(n_levels=3)
print("\n", hierarchy)
for root in hierarchy.roots[:3]:
    print(root.node_id, root.size, root.keywords[:4])
    for child in root.children[:3]:
        print("   ", child.node_id, child.size, child.keywords[:4])
model.visualize_hierarchy_tree().write_html("hierarchy.html")
print("Saved hierarchy.html")
