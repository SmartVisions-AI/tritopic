"""Offline tests for the LLM topic interpreter (the Responses API call is simulated)."""

import numpy as np
import pytest

from tritopic import TriTopic, TriTopicConfig
from tritopic.labeling import TopicInterpreter
from tritopic.labeling.interpreter import SCHEMA

THEMES = {
    "space": "orbit rocket nasa planet galaxy telescope astronaut launch moon satellite",
    "baseball": "pitcher inning homerun batter umpire league season stadium shortstop catcher",
    "medicine": "vaccine patient clinical therapy hospital doctor disease surgery diagnosis treatment",
}
THEME_OF = {w: t for t, v in THEMES.items() for w in v.split()}


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    docs, labels, centers = [], [], rng.normal(size=(3, 32)) * 4
    for t, vocab in enumerate(THEMES.values()):
        for _ in range(40):
            docs.append(" ".join(rng.choice(vocab.split(), 8)))
            labels.append(t)
    labels = np.array(labels)
    emb = centers[labels] + rng.normal(size=(len(docs), 32))
    return docs, labels, emb / np.linalg.norm(emb, axis=1, keepdims=True)


def fake_complete(self, system, user, schema, name):
    """Judge topics like a perfect analyst: mixed if the example docs span >1 theme."""
    self.n_requests += 1
    assert schema is SCHEMA or name == "topic_overview"
    if name == "topic_overview":
        return {"overview": "Three themes.", "groups": [{"name": "All", "topics": ["a"]}]}
    examples = [l for l in user.split("\n") if l.startswith("[")]
    themes = []
    for line in examples:
        words = line.split("] ", 1)[1].split()
        ts = [THEME_OF[w] for w in words if w in THEME_OF]
        themes.append(max(set(ts), key=ts.count))
    distinct = sorted(set(themes))
    if len(distinct) > 1:
        subs = [{"name": t, "description": t, "example_ids": [i for i, x in enumerate(themes) if x == t]} for t in distinct]
        return {"label": " and ".join(distinct), "description": "mix", "aspects": distinct, "verdict": "mixed",
                "sub_themes": subs, "confidence": 0.9, "evidence": "examples split"}
    return {"label": distinct[0].title(), "description": f"About {distinct[0]}.", "aspects": [distinct[0]],
            "verdict": "coherent", "sub_themes": [], "confidence": 0.95, "evidence": "one theme"}


@pytest.fixture
def interpreter(monkeypatch):
    monkeypatch.setattr(TopicInterpreter, "_complete", fake_complete)
    return TopicInterpreter(api_key="test", n_docs=8)


def _model(docs, emb, n_topics):
    cfg = TriTopicConfig(verbose=False, use_dim_reduction=False, n_neighbors=10, auto_resolution=False)
    return TriTopic(config=cfg, n_topics=n_topics).fit(docs, embeddings=emb)


def test_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TopicInterpreter()


def test_sampling_covers_the_spread(corpus):
    docs, labels, emb = corpus
    m = _model(docs, emb, 2)  # one topic must contain two themes
    big = max((t for t in m.topics_ if t.topic_id != -1), key=lambda t: t.size)
    ids = TopicInterpreter.sample_documents(m, big.topic_id, 6)
    assert len(set(labels[ids])) == 2, "proportional sampling should reach both themes"
    assert len(ids) == len(set(ids)) == 6


def test_interpret_labels_and_verdicts(corpus, interpreter):
    docs, labels, emb = corpus
    m = _model(docs, emb, 3)
    res = interpreter.interpret(m)
    assert len(res) == 3 and all(r.verdict == "coherent" for r in res.values())
    assert {t.label for t in m.topics_ if t.topic_id != -1} == {"Space", "Baseball", "Medicine"}
    assert m.interpretations_ is not None and set(m.interpretations_) == set(res)


def test_refine_splits_mixed_topic(corpus, interpreter):
    docs, labels, emb = corpus
    m = _model(docs, emb, 2)
    from sklearn.metrics import adjusted_rand_score
    before = adjusted_rand_score(labels, m.labels_)
    log = interpreter.refine(m)
    assert len(log) == 1 and len(log[0]["sub_themes"]) == 2
    assert adjusted_rand_score(labels, m.labels_) > before
    labels_now = {t.label for t in m.topics_ if t.topic_id != -1}
    assert labels_now == {"Space", "Baseball", "Medicine"}


def test_summarize(corpus, interpreter):
    docs, _, emb = corpus
    m = _model(docs, emb, 3)
    interpreter.interpret(m)
    assert interpreter.summarize(m) == "Three themes."
    assert m.overview_["groups"]
