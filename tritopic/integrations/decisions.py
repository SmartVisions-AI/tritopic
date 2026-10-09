"""
OpenAI Decisions API integration (optional)
===========================================

Typed LLM judgements for TriTopic, via ``POST /v1/decisions``:

1. **Evaluation** -- :func:`word_intrusion` (automated word-intrusion test,
   Chang et al. 2009) and :func:`rate_topics` (0-3 interpretability rating).
2. **Assignment** -- :func:`assign_documents` assigns (new) documents to
   labelled topics with a ``choice`` question; ``TriTopic.reduce_outliers(
   strategy="decisions", decisions_client=...)`` uses it for outliers.
3. **Merging** -- :func:`suggest_merges` asks, for the most similar topic
   pairs, whether both describe the same theme; :func:`apply_merges` merges
   the confirmed pairs.

Nothing here runs inside ``fit()``: clustering stays deterministic and
offline.  Requires an OpenAI API key (``OPENAI_API_KEY``) with access to the
Decisions API (public beta, model ``gpt-6-luna``).  No extra Python package
is needed; ``truststore`` is used for TLS if installed (corporate proxies).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


# ----------------------------------------------------------------------------
# Question builders
# ----------------------------------------------------------------------------

def predicate(name: str, instructions: str) -> dict:
    """Probability that a condition holds."""
    return {"type": "predicate", "name": name, "instructions": instructions}


def choice(name: str, instructions: str, choices: Sequence[tuple[str, str] | str]) -> dict:
    """Pick one value from a fixed list; items are ``value`` or ``(value, description)``."""
    items = [{"value": c} if isinstance(c, str) else {"value": c[0], "description": c[1]} for c in choices]
    return {"type": "choice", "name": name, "instructions": instructions, "choices": items}


def score(name: str, instructions: str, levels: Sequence[tuple[str, str] | str]) -> dict:
    """Rate against ordered levels (lowest first); items are ``label`` or ``(label, description)``."""
    items = [{"label": lv} if isinstance(lv, str) else {"label": lv[0], "description": lv[1]} for lv in levels]
    return {"type": "score", "name": name, "instructions": instructions, "levels": items}


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class DecisionsError(RuntimeError):
    """Raised when the Decisions API returns an error that retries cannot fix."""


@dataclass
class DecisionsClient:
    """
    Minimal client for the OpenAI Decisions API.

    Parameters
    ----------
    api_key : str, optional
        Defaults to the ``OPENAI_API_KEY`` environment variable.
    model : str
        Decision model. Default: ``"gpt-6-luna"``.
    max_workers : int
        Parallel requests in :meth:`decide_many`.
    max_retries : int
        Retries on 429 / 5xx / network errors (exponential backoff).
    cache : bool
        Memoize identical requests in memory (evaluations re-ask a lot).
    """

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    max_retries: int = 4
    max_workers: int = 8
    cache: bool = True
    _cache: dict = field(default_factory=dict, repr=False)
    _ssl: Any = field(default=None, repr=False)
    n_requests: int = 0

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key: pass api_key or set OPENAI_API_KEY.")
        try:
            import truststore
            self._ssl = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            self._ssl = ssl.create_default_context()

    def decide(self, input: str | list, questions: list[dict]) -> dict[str, dict]:
        """Ask *questions* about *input*; returns ``{question name: answer}``."""
        body = {"model": self.model, "input": input, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() if self.cache else None
        if key and key in self._cache:
            return self._cache[key]
        data = self._post(body)
        answers = {a["name"]: a for a in data.get("answers", [])}
        if key:
            self._cache[key] = answers
        return answers

    def decide_many(self, requests: Sequence[tuple[str | list, list[dict]]]) -> list[dict[str, dict]]:
        """Run several :meth:`decide` calls in parallel, preserving order."""
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda r: self.decide(*r), requests))

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/decisions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as resp:
                    self.n_requests += 1
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code not in (408, 409, 429) and e.code < 500 or attempt == self.max_retries:
                    raise DecisionsError(f"Decisions API HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise DecisionsError(f"Decisions API unreachable: {e}") from e
            time.sleep(min(2 ** attempt + random.random(), 30))
        raise DecisionsError("unreachable")  # pragma: no cover


def _answer_value(answer: dict | None, kind: str, default=None):
    if not answer or answer.get("type") == "refusal":
        return default
    return answer.get(kind, default)


def _topic_text(keywords: Sequence[str], docs: Sequence[str] = (), max_doc_chars: int = 300) -> str:
    text = "Keywords: " + ", ".join(keywords)
    for i, d in enumerate(docs, 1):
        text += f"\nExample {i}: " + " ".join(d.split())[:max_doc_chars]
    return text


# ----------------------------------------------------------------------------
# 1. Evaluation
# ----------------------------------------------------------------------------

@dataclass
class IntrusionResult:
    accuracy: float            # share of topics where the intruder was picked
    intruder_probability: float  # mean probability mass on the intruder (lower variance)
    per_topic: list[dict]


def word_intrusion(
    topic_keywords: Sequence[Sequence[str]],
    client: DecisionsClient,
    n_words: int = 5,
    n_candidates_pool: int = 10,
    random_state: int = 42,
) -> IntrusionResult:
    """
    Automated word-intrusion test (Chang et al., 2009).

    For each topic, its top-*n_words* keywords are mixed with one *intruder*:
    a top keyword of another topic that does not appear among this topic's
    top-*n_candidates_pool* keywords.  The LLM has to pick the word that does
    not belong.  Coherent, well-separated topics make the intruder obvious.

    Returns accuracy (hard) and the mean probability assigned to the
    intruder (soft; less noisy for few topics).  Topics with fewer than
    *n_words* keywords are skipped.
    """
    rng = random.Random(random_state)
    # Unique, non-empty keywords in rank order (choice values must be unique;
    # some topic models pad keyword lists with empty strings)
    topics = [list(dict.fromkeys(w for w in kws if w and w.strip())) for kws in topic_keywords]
    requests, meta = [], []
    for i, kws in enumerate(topics):
        if len(kws) < n_words:
            continue
        own = set(kws[:n_candidates_pool])
        others = [j for j in range(len(topics)) if j != i and topics[j]]
        rng.shuffle(others)
        intruder = None
        for j in others:
            cands = [w for w in topics[j][:3] if w not in own]
            if cands:
                intruder = cands[0]
                break
        if intruder is None:
            continue
        words = list(kws[:n_words]) + [intruder]
        rng.shuffle(words)
        q = choice(
            "intruder",
            "These words describe one topic, except one word that was inserted from a different topic. "
            "Pick the word that does not belong.",
            words,
        )
        requests.append((", ".join(words), [q]))
        meta.append({"topic": i, "words": words, "intruder": intruder})

    answers = client.decide_many(requests)
    per_topic = []
    for m, a in zip(meta, answers):
        ans = a.get("intruder")
        picked = _answer_value(ans, "choice")
        probs = {p["value"]: p["probability"] for p in (ans or {}).get("probabilities", [])}
        per_topic.append({**m, "picked": picked, "correct": picked == m["intruder"],
                          "intruder_probability": float(probs.get(m["intruder"], float(picked == m["intruder"])))})
    if not per_topic:
        return IntrusionResult(float("nan"), float("nan"), [])
    return IntrusionResult(
        accuracy=float(np.mean([p["correct"] for p in per_topic])),
        intruder_probability=float(np.mean([p["intruder_probability"] for p in per_topic])),
        per_topic=per_topic,
    )


RATING_LEVELS = [
    ("unrelated", "The words/examples do not share a recognisable theme."),
    ("loose", "Some items are related, but there is no single clear theme."),
    ("mostly coherent", "One theme is recognisable, with a few off-topic items."),
    ("clear", "All items clearly belong to one specific, nameable theme."),
]


def rate_topics(
    topic_keywords: Sequence[Sequence[str]],
    client: DecisionsClient,
    representative_docs: Sequence[Sequence[str]] | None = None,
) -> list[float]:
    """
    Interpretability rating per topic on a 0-3 scale (probability-weighted
    level index; 3 = one clear, nameable theme).  Optionally includes up to
    two representative documents per topic as context.
    """
    q = score("coherence", "How clearly do these keywords (and examples) describe one single, nameable theme?",
              RATING_LEVELS)
    reqs = []
    for i, kws in enumerate(topic_keywords):
        docs = list(representative_docs[i][:2]) if representative_docs else []
        reqs.append((_topic_text(kws[:10], docs), [q]))
    return [float(_answer_value(a.get("coherence"), "score", float("nan"))) for a in client.decide_many(reqs)]


# ----------------------------------------------------------------------------
# 2. Assignment
# ----------------------------------------------------------------------------

def _topic_choices(model, topic_ids: Sequence[int], n_examples: int = 0,
                   example_chars: int = 150) -> list[tuple[str, str]]:
    out = []
    for tid in topic_ids:
        t = model.get_topic(tid)
        desc = (t.label + ": " if t.label else "") + ", ".join(t.keywords[:8])
        if t.description:
            desc += f" ({t.description})"
        for i in t.representative_docs[:n_examples]:
            desc += " | e.g. " + " ".join(model.documents_[i].split())[:example_chars]
        out.append((str(tid), desc[:400 + n_examples * (example_chars + 10)]))
    return out


def assign_documents(
    model,
    documents: Sequence[str],
    client: DecisionsClient,
    embeddings: np.ndarray | None = None,
    n_candidates: int = 10,
    min_confidence: float = 0.0,
    max_doc_chars: int = 2000,
    allow_other: bool = False,
    n_examples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign documents to the fitted model's topics with a ``choice`` question.

    Each topic is offered as its label (if set via ``generate_labels``) plus
    top keywords and example snippets.  With more than
    *n_candidates* topics, the candidates per document are the
    *n_candidates* nearest topic centroids (needs document *embeddings*;
    computed with the model's embedding engine if not given).

    *allow_other* adds an "other" option (abstain -> -1); *n_examples* adds
    that many representative document snippets to each topic description.
    Benchmark (held-out docs, 4 datasets): without "other" and with 2
    examples the LLM matches/slightly beats centroid assignment (0.631 vs.
    0.620); with "other" it abstains on ~35% but is 75% accurate on the rest
    -- the better choice when an uncertain assignment is worse than none
    (``reduce_outliers`` uses it).

    Returns ``(labels, confidence)``; answers below *min_confidence* become -1.
    """
    topic_ids = [t.topic_id for t in model.topics_ if t.topic_id != -1]
    if not topic_ids:
        raise ValueError("Model has no topics.")
    shortlist = None
    if len(topic_ids) > n_candidates:
        if embeddings is None:
            embeddings = model._embedding_engine.encode(list(documents))
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(embeddings, model.topic_embeddings_)
        shortlist = np.argsort(-sims, axis=1)[:, :n_candidates]

    instructions = "Assign the document to the topic it is mainly about."
    if allow_other:
        instructions += " Choose 'other' if none of the topics fits."
    reqs = []
    for i, doc in enumerate(documents):
        cand = topic_ids if shortlist is None else [topic_ids[j] for j in shortlist[i]]
        options = _topic_choices(model, cand, n_examples=n_examples)
        if allow_other:
            options.append(("other", "None of the listed topics fits."))
        reqs.append((" ".join(doc.split())[:max_doc_chars], [choice("topic", instructions, options)]))

    labels = np.full(len(documents), -1, dtype=int)
    conf = np.zeros(len(documents))
    for i, a in enumerate(client.decide_many(reqs)):
        ans = a.get("topic")
        picked = _answer_value(ans, "choice")
        c = float(_answer_value(ans, "confidence", 0.0) or 0.0)
        conf[i] = c
        if picked not in (None, "other") and c >= min_confidence:
            labels[i] = int(picked)
    return labels, conf


# ----------------------------------------------------------------------------
# 3. Merging
# ----------------------------------------------------------------------------

def suggest_merges(
    model,
    client: DecisionsClient,
    n_pairs: int = 15,
    threshold: float = 0.5,
    n_docs: int = 2,
) -> list[tuple[int, int, float]]:
    """
    Ask whether the *n_pairs* most similar topic pairs (centroid cosine)
    describe the same theme.  Returns ``(topic_a, topic_b, probability)`` for
    pairs with probability >= *threshold*, most confident first.

    The judgement is conservative: on the benchmarks it separates same-class
    from different-class pairs better than centroid similarity (AUC
    0.63-0.75 vs. 0.61-0.65) and merges at threshold 0.3-0.5 never hurt NMI,
    but they rarely change it by more than 0.01.  Use it to clean up obvious
    duplicates, not to reach a target topic count (use ``reduce_topics``).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    topics = [t for t in model.topics_ if t.topic_id != -1]
    if len(topics) < 2:
        return []
    sim = cosine_similarity(model.topic_embeddings_)
    iu = np.triu_indices(len(topics), k=1)
    order = np.argsort(-sim[iu])[:n_pairs]
    q = predicate("same_theme", "Do topic A and topic B describe the same underlying theme, so that a "
                                "reader would not want them as separate topics?")
    reqs, pairs = [], []
    for o in order:
        a, b = topics[iu[0][o]], topics[iu[1][o]]
        da = [model.documents_[i] for i in a.representative_docs[:n_docs]]
        db = [model.documents_[i] for i in b.representative_docs[:n_docs]]
        text = "TOPIC A\n" + _topic_text(a.keywords[:10], da) + "\n\nTOPIC B\n" + _topic_text(b.keywords[:10], db)
        reqs.append((text, [q]))
        pairs.append((a.topic_id, b.topic_id))
    out = []
    for (a, b), ans in zip(pairs, client.decide_many(reqs)):
        p = _answer_value(ans.get("same_theme"), "probability")
        if p is not None and p >= threshold:
            out.append((a, b, float(p)))
    return sorted(out, key=lambda x: -x[2])


def apply_merges(model, merges: Sequence[tuple[int, int, float]]):
    """Merge confirmed pairs (transitively: A~B and B~C merge A, B, C)."""
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in merges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    verbose = model.config.verbose
    model.config.verbose = False
    try:
        for members in groups.values():
            if len(members) > 1:
                model.merge_topics(sorted(members))
    finally:
        model.config.verbose = verbose
    return model
