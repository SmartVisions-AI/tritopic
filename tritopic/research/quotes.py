"""Quotable sentences that illustrate a topic."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str, min_words: int, max_words: int) -> list[str]:
    """
    Sentence split that also works for lower-cased or OCR text: split after
    . ! ? regardless of the next character, glue very short fragments (e.g.
    after "mr.") to the next one, and fall back to ~30-word chunks for text
    without sentence punctuation.
    """
    text = " ".join(text.split())
    parts, buf = [], ""
    for frag in _SPLIT.split(text):
        buf = f"{buf} {frag}".strip() if buf else frag
        if len(buf.split()) >= 4:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    out = []
    for p in parts:
        words = p.split()
        if len(words) > max_words:  # no usable punctuation: chunk
            out += [" ".join(words[i:i + 30]) for i in range(0, len(words), 30)]
        else:
            out.append(p)
    return [p for p in out if min_words <= len(p.split()) <= max_words]


def topic_quotes(
    model,
    topic_id: int | None = None,
    n: int = 3,
    max_docs: int = 25,
    min_words: int = 8,
    max_words: int = 45,
    diversity: float = 0.85,
) -> pd.DataFrame:
    """
    The most telling *sentences* of a topic, ready to cite.

    Candidate sentences come from the topic's *max_docs* most central documents.
    They are embedded with the model's embedding engine and ranked by
    similarity to the topic (centroid of those documents in the same
    embedding space) plus a small bonus for topic keywords.  Near-duplicates
    (similarity > *diversity*) and quotes from the same document are skipped.

    Returns ``topic``, ``rank``, ``quote``, ``doc``, ``score``.  With
    ``topic_id=None`` all topics are processed.
    """
    ids = [topic_id] if topic_id is not None else [t.topic_id for t in model.topics_ if t.topic_id != -1]
    labels = np.asarray(model.labels_)
    base = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    rows = []
    for tid in ids:
        members = np.where(labels == tid)[0]
        if len(members) == 0:
            continue
        cen = base[members].mean(axis=0)
        dist = np.linalg.norm(base[members] - cen, axis=1)
        central = members[np.argsort(dist)[:max_docs]]
        cands, owners = [], []
        for d in central:
            for s in _sentences(model.documents_[d], min_words, max_words):
                cands.append(s)
                owners.append(int(d))
        if not cands:
            continue
        S = np.asarray(model._embedding_engine.encode(cands))
        D = np.asarray(model._embedding_engine.encode([model.documents_[d] for d in central]))
        S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
        c = D.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-12
        kws = set(model.get_topic(tid).keywords[:10])
        bonus = np.array([sum(w in s.lower() for w in kws) / max(len(kws), 1) for s in cands])
        score = S @ c + 0.2 * bonus
        chosen, used_docs = [], set()
        for i in np.argsort(-score):
            if owners[i] in used_docs or any(S[i] @ S[j] > diversity for j in chosen):
                continue
            chosen.append(i)
            used_docs.add(owners[i])
            if len(chosen) == n:
                break
        for r, i in enumerate(chosen, 1):
            rows.append(dict(topic=tid, rank=r, quote=cands[i], doc=owners[i], score=float(score[i])))
    return pd.DataFrame(rows, columns=["topic", "rank", "quote", "doc", "score"])
