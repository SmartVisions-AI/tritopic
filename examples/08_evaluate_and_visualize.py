"""
08 - Evaluate a model and create every visualization.

Run:  python examples/08_evaluate_and_visualize.py
All figures are Plotly figures: .show() in a notebook, .write_html() / .write_image() to save.
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tritopic import TopicVisualizer, TriTopic
from tritopic.utils.metrics import compute_silhouette

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.sport.baseball", "sci.med", "comp.graphics", "talk.politics.guns"])
keep = [i for i, d in enumerate(data.data) if len(d.strip()) > 50]
documents = [data.data[i] for i in keep]
true_labels = data.target[keep]

model = TriTopic(verbose=False).fit(documents)

# --- 1. Built-in quality metrics -------------------------------------------------------
metrics = model.evaluate()          # also stores per-topic coherence in get_topic_info()
for name, value in metrics.items():
    print(f"{name:16s} {value}")
print(model.get_topic_info()[["Topic", "Size", "Coherence"]].head().to_string(index=False))

# --- 2. If you have reference labels -----------------------------------------------------
print("NMI vs. newsgroups:", round(normalized_mutual_info_score(true_labels, model.labels_), 3))
print("ARI vs. newsgroups:", round(adjusted_rand_score(true_labels, model.labels_), 3))
print("Silhouette:", round(compute_silhouette(model.original_embeddings_, model.labels_), 3))

# --- 3. Visualizations ---------------------------------------------------------------------
model.visualize(title="Document map").write_html("viz_map.html")              # 2D map of documents
model.visualize_topics(n_keywords=8).write_html("viz_keywords.html")           # keyword bars per topic
model.visualize_hierarchy().write_html("viz_dendrogram.html")                  # topic dendrogram
model.visualize_overlap(threshold=0.1).write_html("viz_overlap.html")         # topic co-occurrence
TopicVisualizer().plot_topic_similarity(model.topic_embeddings_, model.topics_).write_html("viz_similarity.html")
model.build_hierarchy(n_levels=3)
model.visualize_hierarchy_tree().write_html("viz_tree.html")                   # coarse -> fine tree
print("Saved viz_*.html")
