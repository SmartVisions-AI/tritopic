"""Documents and connections between topics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _topic_weight_shares(model, k: int | None = None) -> tuple[np.ndarray, list[int]]:
    """
    Per document: similarity-weighted share of its k nearest neighbours (in the
    unrefined embedding space) that belong to each topic.

    The directed kNN neighbourhood is used on purpose: the clustering graph is
    mutual-kNN, which drops exactly the one-sided links a bridge document has
    into a neighbouring topic.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(model.labels_)
    ids = [t.topic_id for t in model.topics_ if t.topic_id != -1]
    emb = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    k = min(k or model.config.n_neighbors, len(labels) - 1)
    dist, nbr = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb).kneighbors(emb)
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    weights = np.clip(1 - dist, 0, None)
    W = np.zeros((len(labels), len(ids)))
    for j, t in enumerate(ids):
        W[:, j] = (weights * (labels[nbr] == t)).sum(axis=1)
    total = W.sum(axis=1, keepdims=True)
    return np.divide(W, total, out=np.zeros_like(W), where=total > 0), ids


def bridge_documents(model, top_n: int = 20, min_share: float = 0.25) -> pd.DataFrame:
    """
    Documents that connect two topics.

    For every document, its nearest neighbours (embedding space) are counted
    per topic, weighted by similarity.  A bridge document has a substantial
    share (>= *min_share*) of its neighbourhood in a topic other than its own.  ``bridge_score`` is that
    share; documents are returned strongest first, with the two topics and a
    text snippet.  Useful for hybrid cases, interdisciplinary work, and for
    seeing *how* two themes are linked.
    """
    shares, ids = _topic_weight_shares(model)
    labels = np.asarray(model.labels_)
    col = {t: j for j, t in enumerate(ids)}
    rows = []
    for i, l in enumerate(labels):
        if l not in col:
            continue
        s = shares[i].copy()
        own = s[col[l]]
        s[col[l]] = -1
        other = int(np.argmax(s))
        if s[other] >= min_share:
            rows.append(dict(doc=i, topic=int(l), other_topic=ids[other], own_share=float(own),
                             bridge_score=float(s[other]),
                             text=" ".join(model.documents_[i].split())[:200]))
    df = pd.DataFrame(rows, columns=["doc", "topic", "other_topic", "own_share", "bridge_score", "text"])
    return df.sort_values("bridge_score", ascending=False).head(top_n).reset_index(drop=True)


def topic_connections(model) -> pd.DataFrame:
    """
    How strongly topics are linked: normalized association
    ``W_ab / sqrt(W_a · W_b)`` of the nearest-neighbour weight between two
    topics, plus the number of bridge documents between them.  Returned as
    a long DataFrame sorted by strength.
    """
    shares, ids = _topic_weight_shares(model)
    labels = np.asarray(model.labels_)
    col = {t: j for j, t in enumerate(ids)}
    k = len(ids)
    W = np.zeros((k, k))
    for i, l in enumerate(labels):
        if l in col:
            W[col[l]] += shares[i]
    W = (W + W.T) / 2
    vol = W.sum(axis=1)
    bridges = bridge_documents(model, top_n=len(labels))
    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            strength = W[a, b] / np.sqrt(vol[a] * vol[b]) if vol[a] and vol[b] else 0.0
            n_bridge = int(((bridges.topic == ids[a]) & (bridges.other_topic == ids[b])).sum()
                           + ((bridges.topic == ids[b]) & (bridges.other_topic == ids[a])).sum())
            rows.append(dict(topic_a=ids[a], topic_b=ids[b], strength=float(strength), bridge_docs=n_bridge))
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
