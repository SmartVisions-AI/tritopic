"""Tests for seeded topic modeling and the research toolkit (synthetic data, no downloads, no API)."""

import numpy as np
import pandas as pd
import pytest

from tritopic import TriTopic, TriTopicConfig
from tritopic.research import (bridge_documents, compare_groups, distinctive_keywords, methods_report,
                               saturation_curve, topic_connections, topic_evolution, topic_prevalence,
                               topic_quotes, topic_reliability)

THEMES = {
    "space": "orbit rocket nasa planet galaxy telescope astronaut launch moon satellite",
    "baseball": "pitcher inning homerun batter umpire league season stadium shortstop catcher",
    "medicine": "vaccine patient clinical therapy hospital doctor disease surgery diagnosis treatment",
}
NAMES = list(THEMES)
RNG = np.random.default_rng(0)
CENTERS = RNG.normal(size=(3, 32)) * 4
WORD_THEME = {w: i for i, v in enumerate(THEMES.values()) for w in v.split()}


def fake_encode(texts):
    """Embedding = mean of theme centres of the theme words in the text (deterministic)."""
    out = []
    for t in texts:
        ids = [WORD_THEME[w.strip(".,").lower()] for w in t.split() if w.strip(".,").lower() in WORD_THEME]
        v = CENTERS[ids].mean(axis=0) if ids else np.zeros(32)
        out.append(v + 0.01)
    out = np.array(out)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


@pytest.fixture(scope="module")
def corpus():
    """150 docs, 50 per theme; theme 2 only in the second half of the year; docs 0-4 mix themes 0 and 1."""
    rng = np.random.default_rng(1)
    docs, labels, dates = [], [], []
    for t, vocab in enumerate(THEMES.values()):
        words = vocab.split()
        for i in range(50):
            sents = [" ".join(rng.choice(words, 10)).capitalize() + "." for _ in range(3)]
            docs.append(" ".join(sents))
            labels.append(t)
            month = rng.integers(7, 13) if t == 2 else rng.integers(1, 13)
            dates.append(pd.Timestamp(2024, int(month), 1))
    for i in range(5):  # bridge documents
        docs[i] = " ".join(np.random.default_rng(i).choice(THEMES["space"].split() + THEMES["baseball"].split(), 20)) + "."
    labels = np.array(labels)
    emb = fake_encode(docs) + rng.normal(size=(len(docs), 32)) * 0.05
    return docs, labels, emb, dates


def _cfg(**kw):
    base = dict(verbose=False, use_dim_reduction=False, n_neighbors=10, n_consensus_runs=4)
    base.update(kw)
    return TriTopicConfig(**base)


@pytest.fixture(scope="module")
def model(corpus):
    docs, _, emb, _ = corpus
    m = TriTopic(config=_cfg(), n_topics=3).fit(docs, embeddings=emb)
    m._embedding_engine.encode = fake_encode
    return m


# --- seeds -----------------------------------------------------------------------

def test_seeds_codebook_mode(corpus):
    docs, labels, emb, _ = corpus
    seeds = {"Spaceflight": ["orbit", "rocket", "nasa"], "Medicine": "patients in hospital getting therapy"}
    seed_emb = fake_encode([" ".join(seeds["Spaceflight"]), seeds["Medicine"]])
    m = TriTopic(config=_cfg(auto_resolution=False, resolution=0.05)).fit(docs, embeddings=emb, seeds=seeds,
                                                                          seed_embeddings=seed_emb)
    assert set(m.seed_topics_) == {"Spaceflight", "Medicine"}
    space = m.labels_ == m.seed_topics_["Spaceflight"]
    med = m.labels_ == m.seed_topics_["Medicine"]
    assert np.mean(labels[space] == 0) > 0.9 and np.mean(labels[med] == 2) > 0.9
    # the unseeded theme (baseball) emerges as its own topic
    emergent = [t for t in m.emergent_topics_ if np.mean(labels[m.labels_ == t] == 1) > 0.8]
    assert emergent
    info = m.get_topic_info()
    assert set(info.Seed.dropna()) == {"Spaceflight", "Medicine"}
    for name, anchor_ids in m.seed_anchors_.items():
        assert anchor_ids and np.all(m.labels_[anchor_ids] == m.seed_topics_[name])


def test_seed_embedding_dimension_check(corpus):
    docs, _, emb, _ = corpus
    with pytest.raises(ValueError, match="dimensions"):
        TriTopic(config=_cfg()).fit(docs, embeddings=emb, seeds={"x": "orbit"}, seed_embeddings=np.ones((1, 7)))


def test_calibrated_outlier_threshold(model, corpus):
    docs, _, _, _ = corpus
    assert 0 < model.outlier_threshold_ < 1
    assert np.all(model.transform(["orbit rocket nasa planet telescope launch"]) != -1)


# --- reliability, saturation -----------------------------------------------------------

def test_reliability_consensus_and_bootstrap(model):
    rel = topic_reliability(model, method="consensus")
    assert set(rel.columns) >= {"topic", "reliability", "core_share", "reliable"}
    assert rel.reliability.between(0, 1).all() and rel.reliability.min() > 0.8
    assert model.document_stability_.shape == (len(model.documents_),)
    boot = topic_reliability(model, method="bootstrap", n_boot=2)
    assert boot.reliability.min() > 0.6


def test_saturation_curve(model):
    res = saturation_curve(model, fractions=(0.4, 0.7, 1.0), n_repeats=1)
    assert list(res.summary.fraction) == [0.4, 0.7, 1.0]
    assert res.summary.recovered.iloc[-1] == 1.0
    assert res.reference_fraction == 0.7 and 0 <= res.novelty <= 1
    assert res.saturated == (res.novelty <= 0.05)
    assert set(res.late_topics) <= {t.topic_id for t in model.topics_}
    assert "Saturation" in str(res)
    # exact accumulation curve: monotone in the corpus share
    assert res.summary.recovered.is_monotonic_increasing
    refit = saturation_curve(model, fractions=(0.5, 1.0), n_repeats=1, method="refit")
    assert refit.summary.recovered.iloc[-1] == 1.0


# --- bridges, prevalence, groups ---------------------------------------------------------

def test_bridge_documents(model):
    bridges = bridge_documents(model, top_n=10, min_share=0.2)
    assert len(bridges) and set(bridges.doc) & set(range(5)), "the mixed documents should be bridges"
    conn = topic_connections(model)
    assert len(conn) == 3 and conn.strength.between(0, 1).all()


def test_prevalence_wilson_and_bootstrap(model):
    prev = topic_prevalence(model)
    assert np.isclose(prev.share.sum(), 1.0)
    assert (prev.ci_low <= prev.share).all() and (prev.share <= prev.ci_high).all()
    groups = np.where(np.arange(len(model.labels_)) % 2, "a", "b")
    by = topic_prevalence(model, groups=groups, method="bootstrap", n_boot=200)
    assert set(by.group) == {"a", "b"}


def test_compare_groups_detects_real_difference(model, corpus):
    _, labels, _, _ = corpus
    related = np.where(labels == 0, "x", "y")              # group = space or not
    res = compare_groups(model, related)
    space_topic = pd.Series(model.labels_[labels == 0]).mode()[0]
    assert bool(res.set_index("topic").loc[space_topic, "significant"])
    random_groups = np.random.default_rng(3).choice(["x", "y"], len(labels))
    assert not compare_groups(model, random_groups).significant.any()
    words = distinctive_keywords(model, related, "y", "x", n=5)
    assert set(words[words.typical_for == "x"].word) <= set(THEMES["space"].split()) | {"space"}


# --- evolution, quotes, report ------------------------------------------------------------

def test_topic_evolution_finds_birth(model, corpus):
    _, labels, _, dates = corpus
    evo = topic_evolution(model, dates, freq="Q", link_threshold=0.7)
    medicine = pd.Series(model.labels_[labels == 2]).mode()[0]
    lineage = evo.lineage(medicine)
    assert set(lineage.period) <= {"2024Q3", "2024Q4"}, "medicine only exists in the second half"
    births = evo.events[(evo.events.event == "birth")]
    assert any(n in set(lineage.node) for n in births.node)
    # every period topic lists its documents, all from its own period
    quarters = pd.to_datetime(pd.Series(dates)).dt.to_period("Q").astype(str).to_numpy()
    for r in evo.nodes.itertuples():
        assert len(r.docs) == r.size and set(quarters[r.docs]) == {r.period}
    with pytest.raises(ValueError):
        topic_evolution(model, dates[:-1])


def test_topic_quotes(model, corpus):
    _, labels, _, _ = corpus
    q = topic_quotes(model, n=2)
    assert len(q) and q.groupby("topic").size().max() <= 2
    space_topic = pd.Series(model.labels_[labels == 0]).mode()[0]
    quote = q[q.topic == space_topic].quote.iloc[0].lower()
    assert any(w in quote for w in THEMES["space"].split())


def test_methods_report(model):
    text = methods_report(model, corpus="the test corpus")
    assert "150 documents" in text and "consensus Leiden" in text and "| Random seed |" in text


# --- LLM-based: codebook and second coder (simulated) ---------------------------------------

def test_codebook_with_fake_llm(model, monkeypatch):
    from tritopic.labeling import TopicInterpreter

    def fake(self, system, user, schema, name):
        return {"name": "N", "definition": "D", "inclusion": ["i"], "exclusion": ["e"], "coding_notes": "c"}

    monkeypatch.setattr(TopicInterpreter, "_complete", fake)
    cb = TopicInterpreter(api_key="x").codebook(model, n_quotes=2)
    assert len(cb) == 3 and cb.anchor_examples.map(len).max() <= 2
    assert model.codebook_ is cb


def test_intercoder_reliability(model, corpus):
    from tests.test_decisions import FakeClient
    from tritopic.integrations.decisions import intercoder_reliability

    for t in model.topics_:
        if t.topic_id != -1:
            t.label = None
    res = intercoder_reliability(model, FakeClient(), sample_size=60)
    assert res.n == 60 and -1 <= res.kappa <= 1
    assert res.kappa > 0.8, "the fake coder knows the themes"
    assert set(res.per_topic.columns) >= {"precision", "recall", "f1"}


def test_sentence_split_lowercase_and_unpunctuated():
    from tritopic.research.quotes import _sentences
    text = ("iraqi voters turn to economic issues beyond the security situation in iraq today. "
            "a vicious cycle of unemployment and poverty has been made worse by the long war. mr. blair spoke. "
            + "word " * 70)
    out = _sentences(text, 8, 45)
    assert out[0].startswith("iraqi voters") and out[1].startswith("a vicious cycle")
    assert all(8 <= len(s.split()) <= 45 for s in out) and len(out) >= 4
