"""Tests for the structure analyses and the research charts (synthetic data, no downloads, no API)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests.test_research import THEMES, _cfg, corpus, fake_encode, model  # noqa: F401  (fixtures)
from tritopic import TriTopic
from tritopic.labeling.interpreter import TopicInterpretation
from tritopic.research import (codebook_coverage, coassignment, keyword_coverage, resolution_ladder, topic_cores,
                               topic_discovery, topic_evolution, topic_reliability, view_composition)
from tritopic.visualization import charts


# --- structure analyses ---------------------------------------------------------------------

def test_view_composition(model):
    tops, docs = view_composition(model)
    assert len(tops) == 3
    assert np.allclose(tops[["sem_share", "lex_share", "both_share"]].sum(axis=1), 1)
    assert (docs[["sem_only", "lex_only", "both"]] >= 0).all().all()


def test_resolution_ladder_gets_finer(model):
    lad = resolution_ladder(model, multipliers=(0.2, 1, 5))
    counts = lad.levels.groupby("level").size()
    assert counts.iloc[0] <= counts.iloc[-1]
    # flows conserve documents between consecutive levels
    assert lad.flows[lad.flows.level == 0]["count"].sum() <= len(model.labels_)


def test_topic_cores_and_keyword_coverage(model):
    C = topic_cores(model)
    assert C.core_rank.between(0, 1).all() and len(C) == (np.asarray(model.labels_) != -1).sum()
    K = keyword_coverage(model, n_keywords=5)
    assert K.coverage.between(0, 1).all()
    # keywords of a clean synthetic topic are dense inside and rare outside
    assert (K.coverage > K.outside).mean() > 0.8
    assert all(len(p) == model.get_topic(t).size for t, p in zip(K.topic, K.presence))


def test_coassignment_blocks(model):
    res = coassignment(model, sample_size=45, n_refits=3)
    assert res.matrix.shape == (len(res.docs), len(res.docs))
    same = res.topics[:, None] == res.topics[None, :]
    assert res.matrix[same].mean() > res.matrix[~same].mean()


def test_topic_discovery(model):
    T, curve = topic_discovery(model)
    assert (T.f50 <= T.f95).all()
    assert curve.recovered.is_monotonic_increasing and curve.recovered.iloc[-1] > 0.99


def test_codebook_coverage(corpus):
    docs, _, emb, _ = corpus
    m = TriTopic(config=_cfg(auto_resolution=False, resolution=0.05)).fit(
        docs, embeddings=emb, seeds={"Space": ["orbit", "rocket"]}, seed_embeddings=fake_encode(["orbit rocket"]))
    cov = codebook_coverage(m)
    assert abs(cov.share.sum() - (np.asarray(m.labels_) != -1).mean()) < 1e-9
    assert (cov.seed == "Space").any() and cov.seed.isna().any()


# --- charts ------------------------------------------------------------------------------------

def _is_fig(f):
    assert isinstance(f, go.Figure) and len(f.data) + len(f.layout.annotations) > 0


def test_model_charts(model, corpus):
    _, labels, _, dates = corpus
    rel = topic_reliability(model, method="consensus")
    t0 = int(rel.topic.iloc[0])
    _is_fig(charts.plot_topic_table(model, reliability=rel))
    _is_fig(charts.plot_triview(model))
    _is_fig(charts.plot_resolution_ladder(model, multipliers=(0.2, 1, 5)))
    _is_fig(charts.plot_topic_onions(model))
    _is_fig(charts.plot_keyword_barcode(model, t0))
    _is_fig(charts.plot_constellation(model))
    _is_fig(charts.plot_coassignment(model, sample_size=30, n_refits=2))
    _is_fig(charts.plot_group_tilt(model, np.where(np.arange(len(labels)) % 2, "a", "b")))
    _is_fig(charts.plot_trust(model, rel))
    _is_fig(charts.plot_composition(model, labels))
    _is_fig(charts.plot_quote_wall(model))
    _is_fig(charts.plot_topic_discovery(model))
    evo = topic_evolution(model, dates, freq="Q", link_threshold=0.7)
    _is_fig(charts.plot_birth_timeline(evo, highlight={"medicine": "vaccine|patient"}))


def test_seed_and_llm_charts(model, corpus):
    docs, _, emb, _ = corpus
    m = TriTopic(config=_cfg(auto_resolution=False, resolution=0.05)).fit(
        docs, embeddings=emb, seeds={"Space": ["orbit", "rocket"]}, seed_embeddings=fake_encode(["orbit rocket"]))
    fig = charts.plot_codebook_coverage({"one seed": m})
    _is_fig(fig)
    assert "nan" not in " ".join(str(b.text) for b in fig.data)
    assert any(b.marker.pattern.shape == "/" for b in fig.data)  # emerged topics are hatched
    results = {t.topic_id: TopicInterpretation(topic_id=t.topic_id, label=f"Topic {t.topic_id}", description="d", aspects=[],
                                               verdict="mixed" if i == 0 else "coherent",
                                               sub_themes=[{"name": "a"}, {"name": "b"}] if i == 0 else [],
                                               confidence=0.9, evidence="because", size=t.size)
               for i, t in enumerate(t for t in model.topics_ if t.topic_id != -1)}
    first = next(iter(results))
    _is_fig(charts.plot_verdict_board(results, refine_log=[{"topic_id": first, "kept": True}]))

    from tests.test_decisions import FakeClient
    from tritopic.integrations.decisions import intercoder_reliability
    ic = intercoder_reliability(model, FakeClient(), sample_size=40)
    _is_fig(charts.plot_coder_confusion(ic, model))
