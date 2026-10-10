"""Structure of a fitted model: views, granularity, cores, keyword coverage, co-assignment, discovery."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _graph(model):
    """The fitted graph; rebuilt in the model's layout after ``load()`` (which does not store it)."""
    if model.graph_ is not None:
        return model.graph_
    gb = model._graph_builder
    gb.metric = model.config.reduced_metric if model.reduced_embeddings_ is not None else model.config.metric
    model.graph_ = gb.build_multiview_graph(
        semantic_embeddings=model.reduced_embeddings_ if model.reduced_embeddings_ is not None else model.embeddings_,
        lexical_matrix=model.lexical_matrix_ if model.config.use_lexical_view else None,
        weights={"semantic": model.config.semantic_weight, "lexical": model.config.lexical_weight, "metadata": 0.0},
    )
    return model.graph_


def _base_embeddings(model) -> np.ndarray:
    return model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_


def _topic_ids(model) -> list[int]:
    """Non-outlier topic ids, largest first."""
    return [t.topic_id for t in sorted(model.topics_, key=lambda t: -t.size) if t.topic_id != -1]


# --------------------------------------------------------------------------------------------
def view_composition(model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    What holds each topic together: meaning, wording, or both?

    For the links *inside* a topic, counts the links that exist only in the
    semantic view (nearest neighbours by sentence embedding), only in the
    lexical view (TF-IDF neighbours), or in both.  The semantic view is built
    on the unrefined embeddings, so the topics themselves do not shape it.

    Returns ``(topics, documents)``: per topic ``sem_only, lex_only, both``
    counts and shares; per document the same counts for its links inside its
    own topic.  A topic with a high ``sem_only`` share is one theme written in
    different words; a high ``lex_only`` share means shared vocabulary that
    the embeddings do not see.
    """
    if model.lexical_matrix_ is None:
        raise ValueError("view_composition() needs the lexical view (use_lexical_view=True).")
    gb = model._graph_builder
    saved = gb.metric
    gb.metric = "cosine"
    try:
        S = gb.build_hybrid_graph(_base_embeddings(model))
    finally:
        gb.metric = saved
    S = (S.maximum(S.T) > 0).tocsr()
    L = (gb.build_lexical_graph(model.lexical_matrix_) > 0).tocsr()
    L = L.maximum(L.T).tocsr()
    labels = np.asarray(model.labels_)
    trows, drows = [], []
    for t in _topic_ids(model):
        mem = np.where(labels == t)[0]
        s_in, l_in = S[mem][:, mem], L[mem][:, mem]
        both = s_in.multiply(l_in)
        nb, ns, nl = both.nnz, s_in.nnz - both.nnz, l_in.nnz - both.nnz
        tot = max(nb + ns + nl, 1)
        trows.append(dict(topic=t, sem_only=ns, lex_only=nl, both=nb,
                          sem_share=ns / tot, lex_share=nl / tot, both_share=nb / tot))
        bs = np.asarray(both.sum(1)).ravel()
        ss = np.asarray(s_in.sum(1)).ravel() - bs
        ls = np.asarray(l_in.sum(1)).ravel() - bs
        for i, d in enumerate(mem):
            drows.append(dict(doc=int(d), topic=t, sem_only=int(ss[i]), lex_only=int(ls[i]), both=int(bs[i])))
    return pd.DataFrame(trows), pd.DataFrame(drows)


# --------------------------------------------------------------------------------------------
@dataclass
class ResolutionLadder:
    levels: pd.DataFrame   # one row per (level, cluster): multiplier, resolution, size, keywords, main_topic
    flows: pd.DataFrame    # documents flowing from a cluster of level i to a cluster of level i+1
    labels: list           # cluster labels per level (one array per level)


def resolution_ladder(model, multipliers=(0.25, 0.5, 1, 3, 9), n_runs: int = 3) -> ResolutionLadder:
    """
    The same graph clustered at several resolutions (multiples of the
    model's resolution), and how documents flow between the levels.

    Clusters that pass through several levels unchanged are robust themes;
    clusters that split late are sub-themes.  ``main_topic`` is the model
    topic most of a cluster's documents belong to.
    """
    from tritopic.core.clustering import ConsensusLeiden

    graph = _graph(model)
    base = getattr(model, "resolution_", None) or model.config.resolution
    kx = model._keyword_extractor
    kx.fit_corpus(model.documents_)
    topics = np.asarray(model.labels_)
    min_size = max(3, model.config.min_cluster_size)
    rows, all_labels = [], []
    for level, mult in enumerate(multipliers):
        lab = ConsensusLeiden(resolution=base * mult, n_runs=n_runs,
                              random_state=model.config.random_state).fit_predict(graph, min_cluster_size=min_size)
        all_labels.append(lab)
        kw = kx.extract_all_topics(model.documents_, lab, n_keywords=4, method="ctfidf")
        for c in sorted(set(lab.tolist()) - {-1}):
            mask = lab == c
            vals, counts = np.unique(topics[mask], return_counts=True)
            rows.append(dict(level=level, multiplier=mult, resolution=base * mult, cluster=int(c), size=int(mask.sum()),
                             keywords=", ".join(kw[c][0][:3]), main_topic=int(vals[np.argmax(counts)])))
    flows = []
    for level in range(len(all_labels) - 1):
        ct = pd.crosstab(all_labels[level], all_labels[level + 1])
        for a in ct.index:
            for b in ct.columns:
                if a != -1 and b != -1 and ct.loc[a, b] > 0:
                    flows.append(dict(level=level, source=int(a), target=int(b), count=int(ct.loc[a, b])))
    return ResolutionLadder(levels=pd.DataFrame(rows), flows=pd.DataFrame(flows), labels=all_labels)


# --------------------------------------------------------------------------------------------
def topic_cores(model) -> pd.DataFrame:
    """
    Core and edge of every topic, per document.

    ``core_rank`` is 0 for the document closest to its topic centroid and 1
    for the farthest; ``angle`` places documents around the centre by the
    first two principal components of the topic; ``stability`` is the share
    of refits in which the document stayed in its topic (needs
    :func:`topic_reliability` first, else NaN); ``bridge_to`` and
    ``bridge_share`` come from :func:`bridge_documents`.
    """
    from tritopic.research.bridges import bridge_documents

    E = _base_embeddings(model)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(model.labels_)
    stab = getattr(model, "document_stability_", None)
    br = bridge_documents(model, top_n=len(labels), min_share=0.0)
    other = dict(zip(br.doc, br.other_topic))
    share = dict(zip(br.doc, br.bridge_score))
    rows = []
    for t in _topic_ids(model):
        mem = np.where(labels == t)[0]
        c = E[mem].mean(0)
        sim = E[mem] @ (c / (np.linalg.norm(c) + 1e-12))
        rank = (np.argsort(np.argsort(-sim)) + 0.5) / len(mem)
        X = E[mem] - E[mem].mean(0)
        if len(mem) > 2:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            xy = X @ vt[:2].T
        else:
            xy = np.zeros((len(mem), 2))
        ang = np.arctan2(xy[:, 1], xy[:, 0])
        for i, d in enumerate(mem):
            rows.append(dict(doc=int(d), topic=t, core_rank=float(rank[i]), similarity=float(sim[i]), angle=float(ang[i]),
                             stability=float(stab[d]) if stab is not None else np.nan,
                             bridge_to=int(other.get(d, -1)), bridge_share=float(share.get(d, 0.0))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
def keyword_coverage(model, topic_id: int | None = None, n_keywords: int = 10) -> pd.DataFrame:
    """
    Is a keyword carried by the whole topic or by a few documents?

    Per topic keyword: ``coverage`` (share of the topic's documents that
    contain it), ``outside`` (share among all other documents) and
    ``presence`` (0/1 per document of the topic, ordered from the core to the
    edge).  Good keywords are dense inside and rare outside.
    """
    kx = model._keyword_extractor
    dt = kx.fit_corpus(model.documents_)
    vocab = {w: i for i, w in enumerate(kx._vocabulary)}
    E = _base_embeddings(model)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(model.labels_)
    ids = [topic_id] if topic_id is not None else _topic_ids(model)
    rows = []
    for t in ids:
        mem = np.where(labels == t)[0]
        c = E[mem].mean(0)
        order = mem[np.argsort(-(E[mem] @ c))]
        rest = np.where(labels != t)[0]
        for w in model.get_topic(t).keywords[:n_keywords]:
            j = vocab.get(w)
            pres = (dt[order, j] > 0).toarray().ravel().astype(int) if j is not None else np.zeros(len(order), int)
            out = float((dt[rest, j] > 0).mean()) if j is not None and len(rest) else 0.0
            rows.append(dict(topic=t, keyword=w, coverage=float(pres.mean()), outside=out, presence=pres.tolist()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
@dataclass
class CoassignmentResult:
    matrix: np.ndarray     # share of refits (containing both documents) that put them in the same topic
    docs: np.ndarray       # document indices, sorted by topic (largest first) and stability
    topics: np.ndarray     # topic of each document in the fitted model
    n_refits: int


def coassignment(model, sample_size: int = 240, n_refits: int = 10, sample_frac: float = 0.8,
                 random_state: int = 0) -> CoassignmentResult:
    """
    How sharp are the topic borders?  Refits the model on random subsamples
    (same settings and topic count) and counts, for a stratified sample of
    documents, how often two documents land in the same topic.  Solid blocks
    are sharp topics; values between blocks mark documents that switch
    topics when the data changes.
    """
    from tritopic.core.model import TriTopic

    labels = np.asarray(model.labels_)
    n = len(labels)
    ids = _topic_ids(model)
    rng = np.random.default_rng(random_state)
    stab = getattr(model, "document_stability_", None)
    E = _base_embeddings(model)
    sample = []
    for t in ids:
        mem = np.where(labels == t)[0]
        k = min(len(mem), max(4, int(round(sample_size * len(mem) / n))))
        pick = rng.choice(mem, k, replace=False)
        key = -stab[pick] if stab is not None else -(E[pick] @ E[mem].mean(0))
        sample += list(pick[np.argsort(key)])
    sample = np.array(sample)
    runs = []
    for b in range(n_refits):
        idx = np.sort(np.random.default_rng(random_state + 100 + b).choice(n, int(sample_frac * n), replace=False))
        cfg = copy.deepcopy(model.config)
        cfg.verbose = False
        sub = TriTopic(config=cfg, n_topics=len(ids)).fit([model.documents_[i] for i in idx], embeddings=E[idx])
        full = -2 * np.ones(n, dtype=int)
        full[idx] = sub.labels_
        runs.append(full[sample])
    R = np.array(runs)
    present = (R[:, :, None] >= 0) & (R[:, None, :] >= 0)
    same = present & (R[:, :, None] == R[:, None, :])
    M = same.sum(0) / np.maximum(present.sum(0), 1)
    return CoassignmentResult(matrix=M, docs=sample, topics=labels[sample], n_refits=n_refits)


# --------------------------------------------------------------------------------------------
def topic_discovery(model, min_docs: int | None = None, fractions=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    When does each topic become visible while reading the corpus in random order?

    A topic counts as visible once ``min_docs`` (default ``min_cluster_size``)
    of its documents have been read; the probability is hypergeometric.
    Returns ``(topics, curve)``: per topic the share of the corpus after which
    it is visible with 50% (``f50``) and 95% (``f95``) probability; the curve
    gives the expected share of visible topics per fraction (the exact
    accumulation curve of :func:`saturation_curve`).
    """
    from scipy.stats import hypergeom

    labels = np.asarray(model.labels_)
    n = len(labels)
    k = min_docs or max(3, model.config.min_cluster_size)
    fr = np.linspace(0.01, 1, 100) if fractions is None else np.asarray(fractions)
    sizes = {t: int((labels == t).sum()) for t in _topic_ids(model)}
    P = {t: np.array([1 - hypergeom.cdf(k - 1, n, s, int(round(f * n))) for f in fr]) for t, s in sizes.items()}
    rows = []
    for t, p in P.items():
        f50 = float(fr[np.argmax(p >= 0.5)]) if (p >= 0.5).any() else np.nan
        f95 = float(fr[np.argmax(p >= 0.95)]) if (p >= 0.95).any() else np.nan
        rows.append(dict(topic=t, size=sizes[t], f50=f50, f95=f95))
    curve = pd.DataFrame(dict(fraction=fr, recovered=np.mean(list(P.values()), axis=0) if P else np.zeros(len(fr))))
    return pd.DataFrame(rows), curve


# --------------------------------------------------------------------------------------------
def codebook_coverage(model) -> pd.DataFrame:
    """Per topic: size, share of the corpus and the seed it grew from (``None`` = emerged)."""
    n = len(model.labels_)
    rows = [dict(topic=t.topic_id, size=t.size, share=t.size / n, seed=t.seed, keywords=", ".join(t.keywords[:4]))
            for t in sorted(model.topics_, key=lambda t: (t.seed is None, -t.size)) if t.topic_id != -1]
    return pd.DataFrame(rows)
