"""Tests for the OpenAI Decisions API integration (offline, simulated API)."""

import io
import json
import urllib.error

import numpy as np
import pytest

from tritopic import TriTopic, TriTopicConfig
from tritopic.integrations import decisions as dec

THEMES = {
    "space": "orbit rocket nasa planet galaxy telescope astronaut launch moon satellite",
    "baseball": "pitcher inning homerun batter umpire league season stadium shortstop catcher",
    "medicine": "vaccine patient clinical therapy hospital doctor disease surgery diagnosis treatment",
}
THEME_OF = {w: t for t, vocab in THEMES.items() for w in vocab.split()}


class FakeClient:
    """Answers like a perfect judge that knows THEMES."""

    def __init__(self):
        self.requests = []

    def decide_many(self, requests):
        return [self.decide(i, q) for i, q in requests]

    def decide(self, input, questions):
        self.requests.append((input, questions))
        q = questions[0]
        words = [w.strip(",.:") for w in input.lower().split()]
        themes = [THEME_OF[w] for w in words if w in THEME_OF]
        majority = max(set(themes), key=themes.count) if themes else None
        if q["name"] == "intruder":
            values = [c["value"] for c in q["choices"]]
            odd = next(v for v in values if THEME_OF.get(v) != majority)
            probs = [{"value": v, "probability": 0.9 if v == odd else 0.02} for v in values]
            return {"intruder": {"type": "choice", "name": "intruder", "choice": odd, "probabilities": probs,
                                 "confidence": 0.9}}
        if q["name"] == "coherence":
            return {"coherence": {"type": "score", "name": "coherence", "score": 3.0 if len(set(themes)) == 1 else 1.0}}
        if q["name"] == "topic":
            best, best_overlap = "other", 0
            for c in q["choices"]:
                overlap = sum(w.strip(",") in words for w in c.get("description", "").split())
                if overlap > best_overlap:
                    best, best_overlap = c["value"], overlap
            return {"topic": {"type": "choice", "name": "topic", "choice": best, "confidence": 0.95}}
        if q["name"] == "same_theme":
            a, b = input.split("TOPIC B")
            ta = {THEME_OF[w.strip(",.:")] for w in a.lower().split() if w.strip(",.:") in THEME_OF}
            tb = {THEME_OF[w.strip(",.:")] for w in b.lower().split() if w.strip(",.:") in THEME_OF}
            return {"same_theme": {"type": "predicate", "name": "same_theme", "probability": 0.95 if ta == tb else 0.05}}
        raise AssertionError(q)


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    docs, labels, centers = [], [], rng.normal(size=(3, 32)) * 4
    for t, vocab in enumerate(THEMES.values()):
        words = vocab.split()
        for _ in range(40):
            docs.append(" ".join(rng.choice(words, 8)))
            labels.append(t)
    labels = np.array(labels)
    emb = centers[labels] + rng.normal(size=(len(docs), 32))
    return docs, labels, emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _model(docs, emb, resolution=0.05, n_topics="auto"):
    cfg = TriTopicConfig(verbose=False, use_dim_reduction=False, n_neighbors=10,
                         auto_resolution=False, resolution=resolution)
    return TriTopic(config=cfg, n_topics=n_topics).fit(docs, embeddings=emb)


# --- client --------------------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_request_shape_cache_and_retry(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout, context):
        calls.append(json.loads(req.data))
        assert req.full_url.endswith("/v1/decisions")
        assert req.headers["Authorization"] == "Bearer test-key"
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, io.BytesIO(b"slow down"))
        return _Resp(json.dumps({"answers": [{"type": "predicate", "name": "p", "probability": 0.8}]}).encode())

    monkeypatch.setattr(dec.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dec.time, "sleep", lambda s: None)
    client = dec.DecisionsClient(api_key="test-key")
    q = dec.predicate("p", "Is it?")
    out = client.decide("some text", [q])
    assert out["p"]["probability"] == 0.8
    assert calls[-1] == {"model": "gpt-6-luna", "input": "some text", "questions": [q]}
    client.decide("some text", [q])  # cached
    assert len(calls) == 2


def test_client_raises_on_client_error(monkeypatch):
    def fake_urlopen(req, timeout, context):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"invalid question"))

    monkeypatch.setattr(dec.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(dec.DecisionsError, match="400"):
        dec.DecisionsClient(api_key="k").decide("x", [dec.predicate("p", "?")])


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        dec.DecisionsClient()


def test_question_builders():
    c = dec.choice("c", "pick", ["a", ("b", "desc")])
    assert c["choices"] == [{"value": "a"}, {"value": "b", "description": "desc"}]
    s = dec.score("s", "rate", [("low", "x"), "high"])
    assert s["levels"] == [{"label": "low", "description": "x"}, {"label": "high"}]


# --- 1. evaluation --------------------------------------------------------

def test_word_intrusion_perfect_topics():
    topics = [v.split() for v in THEMES.values()]
    res = dec.word_intrusion(topics, FakeClient())
    assert res.accuracy == 1.0
    assert res.intruder_probability == pytest.approx(0.9)
    assert all(p["intruder"] not in topics[p["topic"]][:10] for p in res.per_topic)


def test_word_intrusion_mixed_topic_is_detected_as_worse():
    mixed = ["orbit", "pitcher", "vaccine", "rocket", "inning", "patient"]
    topics = [v.split() for v in THEMES.values()] + [mixed]
    res = dec.word_intrusion(topics, FakeClient())
    assert res.accuracy < 1.0


def test_rate_topics():
    ratings = dec.rate_topics([THEMES["space"].split(), ["orbit", "pitcher", "vaccine"]], FakeClient())
    assert ratings == [3.0, 1.0]


# --- 2. assignment --------------------------------------------------------

def test_assign_documents_matches_themes(corpus):
    docs, labels, emb = corpus
    model = _model(docs, emb, resolution=0.05)
    new_docs = ["the astronaut saw the moon from orbit", "the pitcher struck out the batter",
                "the doctor started a new therapy", "completely unrelated cooking recipe"]
    pred, conf = dec.assign_documents(model, new_docs, FakeClient(), embeddings=emb[:4], allow_other=True)
    space_topic = model.labels_[0]
    assert pred[0] == space_topic
    assert pred[1] == model.labels_[40] and pred[2] == model.labels_[80]
    assert pred[3] == -1  # "other"


def test_assign_documents_shortlists_candidates(corpus):
    docs, _, emb = corpus
    model = _model(docs, emb, n_topics=9)
    n_topics = len([t for t in model.topics_ if t.topic_id != -1])
    assert n_topics > 3
    client = FakeClient()
    dec.assign_documents(model, docs[:3], client, embeddings=emb[:3], n_candidates=3)
    for _, (q,) in client.requests:
        assert len(q["choices"]) == 3  # shortlist, no "other" by default


def test_reduce_outliers_with_decisions(corpus):
    docs, labels, emb = corpus
    model = _model(docs, emb, resolution=0.05)
    model.labels_[[1, 41, 81]] = -1
    model.reduce_outliers(strategy="decisions", decisions_client=FakeClient())
    assert (model.labels_ == -1).sum() == 0
    assert model.labels_[1] == model.labels_[0] and model.labels_[81] == model.labels_[80]
    model.labels_[1] = -1
    with pytest.raises(ValueError):
        model.reduce_outliers(strategy="decisions")


# --- 3. merging -----------------------------------------------------------

def test_suggest_and_apply_merges(corpus):
    docs, _, emb = corpus
    model = _model(docs, emb, n_topics=9)  # over-segmented: several topics per theme
    before = len([t for t in model.topics_ if t.topic_id != -1])
    merges = dec.suggest_merges(model, FakeClient(), n_pairs=50)
    assert merges and all(p >= 0.5 for _, _, p in merges)
    dec.apply_merges(model, merges)
    after = len([t for t in model.topics_ if t.topic_id != -1])
    assert after < before
    assert after >= 3  # never merges across themes
