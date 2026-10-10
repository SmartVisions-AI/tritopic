"""Topic evolution over time: births, deaths, splits and merges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TopicEvolution:
    nodes: pd.DataFrame   # one row per (period, period topic): size, keywords, global topic, docs
    links: pd.DataFrame   # successor links between consecutive periods with similarity
    events: pd.DataFrame  # birth / continuation / split / merge / death per period topic

    def lineage(self, global_topic: int) -> pd.DataFrame:
        """All period topics that belong to one topic of the global model."""
        return self.nodes[self.nodes.global_topic == global_topic]


def topic_evolution(
    model,
    timestamps,
    freq: str = "Q",
    link_threshold: float = 0.75,
    resolution: float | None = None,
    min_topic_size: int | None = None,
    n_runs: int = 3,
) -> TopicEvolution:
    """
    Cluster each time period separately and link topics across periods.

    Documents are grouped by period (pandas period alias *freq*: ``"M"``,
    ``"Q"``, ``"Y"``).  For each period a kNN graph is built from scratch on
    that period's documents (same reduced embedding layout and lexical view as
    the model, so all periods live in one space) and clustered with consensus
    Leiden.  *resolution* defaults to 4x the model's resolution: periods are
    clustered finer than the global model, so that events inside a broad
    global topic (a pandemic inside "health", a war inside "world news")
    become visible.  Topics of consecutive periods are linked when their
    centroids (original embeddings) have cosine similarity >= *link_threshold*.
    From the links:

    * **start**: topic of the first period; **birth**: no predecessor
    * **continuation**: one predecessor that has one successor
    * **split**: the predecessor has several successors
    * **merge**: several predecessors
    * **death**: no successor (not in the last period)

    Every period topic is mapped to the global topic most of its documents
    belong to (``global_topic``); ``docs`` holds its document indices.

    Validated on HuffPost news 2012-2022 (16,400 dated articles): births
    detected the COVID pandemic (2020) and the war in Ukraine (2022) for all
    tested sample sizes and settings, plus events such as the Flint water
    crisis, the Hong Kong protests and Roe v. Wade.  Reusing each period's
    part of the global graph instead (up to 2.5.0) left about 5 of 28 edges
    per document and produced mostly noise (70% of period topics "born").
    Themes that skip a period count as a new birth, and recurring formats can
    occasionally appear as births; read births together with their keywords.
    """
    from tritopic.core.clustering import ConsensusLeiden

    ts = pd.to_datetime(pd.Series(list(timestamps)))
    if len(ts) != len(model.labels_):
        raise ValueError(f"{len(ts)} timestamps for {len(model.labels_)} documents.")
    periods = ts.dt.to_period(freq).to_numpy()
    labels = np.asarray(model.labels_)
    emb = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    layout = model.reduced_embeddings_ if model.reduced_embeddings_ is not None else model.embeddings_
    lexical = model.lexical_matrix_ if model.config.use_lexical_view else None
    res = resolution if resolution is not None else 4 * model.resolution_
    min_size = min_topic_size or max(3, model.config.min_cluster_size)
    gb = model._graph_builder
    saved_metric = gb.metric
    gb.metric = model.config.reduced_metric if model.reduced_embeddings_ is not None else model.config.metric
    kx = model._keyword_extractor
    kx.fit_corpus(model.documents_)

    nodes, cents = [], {}
    try:
        for p in sorted(pd.unique(periods)):
            idx = np.where(periods == p)[0]
            if len(idx) < 2 * min_size:
                continue
            graph = gb.build_multiview_graph(
                semantic_embeddings=layout[idx],
                lexical_matrix=lexical[idx] if lexical is not None else None,
                weights={"semantic": model.config.semantic_weight, "lexical": model.config.lexical_weight,
                         "metadata": 0.0},
            )
            cl = ConsensusLeiden(resolution=res, n_runs=n_runs, random_state=model.config.random_state)
            sub_labels = cl.fit_predict(graph, min_cluster_size=min_size)
            full = -np.ones(len(labels), dtype=int)
            full[idx] = sub_labels
            kw = kx.extract_all_topics(model.documents_, full, n_keywords=8, method="ctfidf")
            for t in sorted(set(sub_labels.tolist()) - {-1}):
                members = idx[sub_labels == t]
                vals, counts = np.unique(labels[members], return_counts=True)
                node = f"{p}:{t}"
                cents[node] = emb[members].mean(axis=0)
                nodes.append(dict(node=node, period=str(p), size=len(members), keywords=", ".join(kw[t][0][:6]),
                                  global_topic=int(vals[np.argmax(counts)]), docs=members.tolist()))
    finally:
        gb.metric = saved_metric
    nodes = pd.DataFrame(nodes, columns=["node", "period", "size", "keywords", "global_topic", "docs"])
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
        events.append(dict(node=r.node, period=r.period, event=kind, keywords=r.keywords, size=r.size,
                           predecessors=preds))
        if r.period != last and out_deg.get(r.node, 0) == 0:
            events.append(dict(node=r.node, period=r.period, event="death", keywords=r.keywords, size=r.size,
                               predecessors=[]))
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
