"""Topic evolution over time: births, deaths, splits and merges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TopicEvolution:
    nodes: pd.DataFrame   # one row per (period, period topic): size, keywords, global topic
    links: pd.DataFrame   # successor links between consecutive periods with similarity
    events: pd.DataFrame  # birth / continuation / split / merge / death per period topic

    def lineage(self, global_topic: int) -> pd.DataFrame:
        """All period topics that belong to one topic of the global model."""
        return self.nodes[self.nodes.global_topic == global_topic]


def topic_evolution(
    model,
    timestamps,
    freq: str = "Q",
    link_threshold: float = 0.8,
    min_topic_size: int | None = None,
    n_runs: int = 3,
) -> TopicEvolution:
    """
    Cluster each time period separately and link topics across periods.

    Documents are grouped by period (pandas period alias *freq*: ``"M"``,
    ``"Q"``, ``"Y"``).  Within each period, the documents' part of the fitted
    graph is clustered with consensus Leiden at the model's resolution.
    Topics of consecutive periods are linked when their centroids have cosine
    similarity >= *link_threshold*.  From the links:

    * **birth**: no predecessor; **death**: no successor (not in the last period)
    * **continuation**: one predecessor that has one successor
    * **split**: the predecessor has several successors
    * **merge**: several predecessors

    Every period topic is also mapped to the global topic most of its
    documents belong to (``global_topic``), so lineages can be read against
    the overall model.  Unlike frequency-over-time plots, this shows how the
    *content* of topics develops.
    """
    from tritopic.core.clustering import ConsensusLeiden

    ts = pd.to_datetime(pd.Series(list(timestamps)))
    periods = ts.dt.to_period(freq)
    labels = np.asarray(model.labels_)
    emb = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    min_size = min_topic_size or max(3, model.config.min_cluster_size)
    kx = model._keyword_extractor
    kx.fit_corpus(model.documents_)

    nodes, cents = [], {}
    order = sorted(periods.unique())
    for p in order:
        idx = np.where(periods.to_numpy() == p)[0]
        if len(idx) < 2 * min_size:
            continue
        sub = model.graph_.subgraph(idx.tolist())
        cl = ConsensusLeiden(resolution=model.resolution_, n_runs=n_runs, random_state=model.config.random_state)
        sub_labels = cl.fit_predict(sub, min_cluster_size=min_size)
        full = -np.ones(len(labels), dtype=int)
        full[idx] = sub_labels
        kw = kx.extract_all_topics(model.documents_, full, n_keywords=8, method="ctfidf")
        for t in sorted(set(sub_labels.tolist()) - {-1}):
            members = idx[sub_labels == t]
            vals, counts = np.unique(labels[members], return_counts=True)
            node = f"{p}:{t}"
            cents[node] = emb[members].mean(axis=0)
            nodes.append(dict(node=node, period=str(p), size=len(members), keywords=", ".join(kw[t][0][:6]),
                              global_topic=int(vals[np.argmax(counts)])))
    nodes = pd.DataFrame(nodes)
    links = []
    periods_present = list(dict.fromkeys(nodes.period)) if len(nodes) else []
    for a, b in zip(periods_present, periods_present[1:]):
        A = nodes[nodes.period == a].node.tolist()
        B = nodes[nodes.period == b].node.tolist()
        for x in A:
            for y in B:
                cx, cy = cents[x], cents[y]
                sim = float(cx @ cy / (np.linalg.norm(cx) * np.linalg.norm(cy) + 1e-12))
                if sim >= link_threshold:
                    links.append(dict(source=x, target=y, similarity=sim))
    links = pd.DataFrame(links, columns=["source", "target", "similarity"])
    out_deg = links.source.value_counts().to_dict()
    in_deg = links.target.value_counts().to_dict()
    last = periods_present[-1] if periods_present else None
    events = []
    for r in nodes.itertuples():
        preds = links[links.target == r.node].source.tolist()
        if not preds:
            kind = "birth" if r.period != periods_present[0] else "start"
        elif len(preds) > 1:
            kind = "merge"
        elif out_deg.get(preds[0], 0) > 1:
            kind = "split"
        else:
            kind = "continuation"
        events.append(dict(node=r.node, period=r.period, event=kind, predecessors=preds))
        if r.period != last and out_deg.get(r.node, 0) == 0:
            events.append(dict(node=r.node, period=r.period, event="death", predecessors=[]))
    return TopicEvolution(nodes=nodes, links=links, events=pd.DataFrame(events))


def plot_evolution(evo: TopicEvolution):
    """Sankey diagram of topics flowing from period to period."""
    import plotly.graph_objects as go

    names = evo.nodes.node.tolist()
    index = {n: i for i, n in enumerate(names)}
    labels = [f"{r.period}: {r.keywords.split(', ')[0]} ({r.size})" for r in evo.nodes.itertuples()]
    size = dict(zip(evo.nodes.node, evo.nodes["size"]))
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=12, thickness=12),
        link=dict(source=[index[s] for s in evo.links.source], target=[index[t] for t in evo.links.target],
                  value=[min(size[s], size[t]) for s, t in zip(evo.links.source, evo.links.target)]),
    ))
    fig.update_layout(title="Topic evolution")
    return fig
