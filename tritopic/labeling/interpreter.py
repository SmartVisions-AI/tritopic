"""
LLM topic interpretation
========================

:class:`TopicInterpreter` lets an LLM read each topic the way an analyst would
and returns a structured interpretation:

* label, description and the main aspects of the topic,
* a **verdict**: one coherent theme, a mix of distinct themes, or unclear,
* for mixed topics, the sub-themes (with the example documents behind each).

The evidence per topic is richer than for plain labelling: ranked keywords,
example documents chosen to *cover the topic in proportion* (one typical
document per k-means region, not only the most typical documents, so a
mixture becomes visible),
and the neighbouring topics for contrast.  :meth:`TopicInterpreter.refine`
uses the verdicts to split mixed topics with ``TriTopic.divide`` and
interprets the new topics again.

Backend: OpenAI Responses API with a JSON schema (strict structured output),
plain HTTP, no extra dependency.  Needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tritopic.integrations.decisions import DecisionsError, post_json, tls_context

# Chosen on the dev splits (20NG test, BBC test, AG News train sample, arXiv validation;
# deliberately coarse topics): gpt-6-luna detected the most mixed topics (refine() NMI
# 0.614 -> 0.654 vs. 0.638 for gpt-5.5 and 0.625 for gpt-5.4-mini, which missed the BBC
# politics/business mixture). With the refine() safeguards (accept="consistency"),
# held-out paper splits: NMI 0.606 -> 0.639, no dataset worse by more than 0.001.
DEFAULT_MODEL = "gpt-6-luna"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "description", "aspects", "verdict", "sub_themes", "confidence", "evidence"],
    "properties": {
        "label": {"type": "string", "description": "2-5 word name of the topic, specific enough to tell it apart from its neighbours"},
        "description": {"type": "string", "description": "1-2 sentences: what the documents in this topic are about"},
        "aspects": {"type": "array", "items": {"type": "string"}, "description": "2-5 recurring aspects or sub-subjects"},
        "verdict": {"type": "string", "enum": ["coherent", "mixed", "unclear"]},
        "sub_themes": {
            "type": "array",
            "description": "Only for verdict 'mixed': the distinct themes, otherwise empty",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "description", "example_ids"],
                "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                               "example_ids": {"type": "array", "items": {"type": "integer"}}},
            },
        },
        "confidence": {"type": "number", "description": "0-1, how sure the verdict is"},
        "evidence": {"type": "string", "description": "One sentence naming the evidence for the verdict"},
    },
}

SYSTEM_PROMPT = """You are an expert analyst who interprets topics found by a topic model.
For one topic you get: its keywords (ranked by distinctiveness), example documents sampled to cover
the whole spread of the topic (not only its most typical documents), and the neighbouring topics.

Decide whether the documents share ONE theme a domain expert would name with a single label
("coherent"), or whether they fall into two or more groups about clearly different subjects that an
expert would file under different headings ("mixed"). Different facets of one subject (e.g. matches
and transfers in football) are still coherent. Use "unclear" only if the documents share no
recognisable theme at all.

Label the topic so that a reader can tell it apart from the neighbouring topics. Base everything on
the documents, not on the keywords alone. Write all text in {language}."""


@dataclass
class TopicInterpretation:
    topic_id: int
    label: str
    description: str
    aspects: list[str]
    verdict: str
    sub_themes: list[dict]
    confidence: float
    evidence: str
    size: int = 0
    example_doc_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TopicInterpreter:
    """
    Interpret topics of a fitted :class:`~tritopic.TriTopic` model with an LLM.

    Parameters
    ----------
    model : str
        OpenAI model for the Responses API. Default ``"gpt-6-luna"``.
    api_key : str, optional
        Defaults to ``OPENAI_API_KEY``.
    language : str
        Output language for labels and descriptions. Default ``"English"``.
    domain_hint : str, optional
        One sentence of context ("BBC news articles", "hotel reviews").
    n_docs : int
        Example documents per topic. Default 8.
    max_doc_chars : int
        Characters per example document. Default 600.
    reasoning_effort : str or None
        ``"low"``, ``"medium"``, ``"high"`` for reasoning models, None to omit.
    max_workers : int
        Topics interpreted in parallel.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        language: str = "English",
        domain_hint: str | None = None,
        n_docs: int = 8,
        max_doc_chars: int = 600,
        reasoning_effort: str | None = "low",
        max_workers: int = 8,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key: pass api_key or set OPENAI_API_KEY.")
        self.language = language
        self.domain_hint = domain_hint
        self.n_docs = n_docs
        self.max_doc_chars = max_doc_chars
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._ssl = tls_context()
        self.n_requests = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------ LLM
    def _complete(self, system: str, user: str, schema: dict, name: str) -> dict:
        body = {
            "model": self.model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        data = post_json(f"{self.base_url}/responses", body, self.api_key, self._ssl,
                         self.timeout, self.max_retries, error_prefix="OpenAI Responses API")
        self.n_requests += 1
        for k in self.usage:
            self.usage[k] += int(data.get("usage", {}).get(k, 0) or 0)
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise DecisionsError(f"Model refused: {part.get('refusal')}")
                    if "text" in part:
                        return json.loads(part["text"])
        raise DecisionsError("No message in response")

    # ------------------------------------------------------------ evidence
    @staticmethod
    def sample_documents(model, topic_id: int, n: int) -> list[int]:
        """
        One typical document per region of the topic: k-means with *n*
        clusters on the topic's embeddings, the document nearest to each
        cluster centre, largest cluster first.  Large sub-groups get
        proportionally more examples (so a real sub-theme is backed by several
        documents) while the spread of the topic is still covered.
        """
        from sklearn.cluster import KMeans

        idx = np.where(model.labels_ == topic_id)[0]
        if len(idx) <= n:
            return idx.tolist()
        emb = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
        E = emb[idx]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=n, n_init=3, random_state=0).fit(E)
        order = np.argsort(-np.bincount(km.labels_, minlength=n))
        chosen = []
        for c in order:
            members = np.where(km.labels_ == c)[0]
            if len(members):
                chosen.append(int(idx[members[np.argmin(np.linalg.norm(E[members] - km.cluster_centers_[c], axis=1))]]))
        return chosen

    def _neighbours(self, model, topic_id: int, k: int = 3) -> list[str]:
        from sklearn.metrics.pairwise import cosine_similarity
        topics = [t for t in model.topics_ if t.topic_id != -1]
        ids = [t.topic_id for t in topics]
        if topic_id not in ids or len(ids) < 2:
            return []
        sims = cosine_similarity(model.topic_embeddings_)[ids.index(topic_id)]
        order = [j for j in np.argsort(-sims) if ids[j] != topic_id][:k]
        out = []
        for j in order:
            t = topics[j]
            name = f'"{t.label}": ' if t.label else ""
            out.append(f"- {name}{', '.join(t.keywords[:8])}")
        return out

    def _prompt(self, model, topic) -> tuple[str, list[int]]:
        doc_ids = self.sample_documents(model, topic.topic_id, self.n_docs)
        n_total = int((model.labels_ != -1).sum()) or len(model.labels_)
        lines = []
        if self.domain_hint:
            lines.append(f"Corpus: {self.domain_hint}")
        lines += [
            f"Topic size: {topic.size} documents ({100 * topic.size / n_total:.1f}% of the corpus)",
            f"Keywords: {', '.join(topic.keywords[:15])}",
            "",
            "Neighbouring topics (for contrast):",
            *(self._neighbours(model, topic.topic_id) or ["- none"]),
            "",
            "Example documents (id: text):",
        ]
        for i, d in enumerate(doc_ids):
            lines.append(f"[{i}] " + " ".join(model.documents_[d].split())[: self.max_doc_chars])
        return "\n".join(lines), doc_ids

    # ------------------------------------------------------------- public
    def interpret_topic(self, model, topic_id: int) -> TopicInterpretation:
        topic = model.get_topic(topic_id)
        user, doc_ids = self._prompt(model, topic)
        r = self._complete(SYSTEM_PROMPT.format(language=self.language), user, SCHEMA, "topic_interpretation")
        subs = []
        for st in r.get("sub_themes", []) if r["verdict"] == "mixed" else []:
            subs.append({**st, "doc_ids": [doc_ids[i] for i in st.get("example_ids", []) if 0 <= i < len(doc_ids)]})
        return TopicInterpretation(
            topic_id=topic_id, label=r["label"].strip(), description=r["description"].strip(),
            aspects=r.get("aspects", []), verdict=r["verdict"], sub_themes=subs,
            confidence=float(r.get("confidence", 0.0)), evidence=r.get("evidence", ""),
            size=topic.size, example_doc_ids=doc_ids,
        )

    def interpret(self, model, topics: list[int] | None = None, apply_labels: bool = True) -> dict[int, TopicInterpretation]:
        """
        Interpret topics (default: all) in parallel.  With *apply_labels*, the
        labels and descriptions are written to ``model.topics_`` (as
        ``generate_labels`` does).  Results are also stored in
        ``model.interpretations_``.
        """
        ids = topics or [t.topic_id for t in model.topics_ if t.topic_id != -1]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = dict(zip(ids, pool.map(lambda tid: self.interpret_topic(model, tid), ids)))
        if apply_labels:
            for tid, r in results.items():
                t = model.get_topic(tid)
                t.label, t.description = r.label, r.description
        store = getattr(model, "interpretations_", None) or {}
        store.update(results)
        model.interpretations_ = store
        return results

    @staticmethod
    def _topic_coherence(model, topic_ids) -> dict[int, float]:
        """NPMI of each topic's top-10 keywords, from the cached document-term matrix."""
        from tritopic.utils.metrics import coherence_from_doc_term

        kx = model._keyword_extractor
        doc_term = kx.fit_corpus(model.documents_)
        index = {w: i for i, w in enumerate(kx._vocabulary)}
        ids = [t for t in topic_ids if model.get_topic(t) is not None]
        terms = [[index[w] for w in model.get_topic(t).keywords[:10] if w in index] for t in ids]
        return dict(zip(ids, coherence_from_doc_term(doc_term, terms)))

    @staticmethod
    def _split_matches_themes(labels: np.ndarray, themes: list[dict], min_agreement: float = 0.6) -> bool:
        """True if each sub-theme's examples mostly share one new topic and the themes differ."""
        homes = []
        for th in themes:
            got = labels[th["doc_ids"]]
            vals, counts = np.unique(got, return_counts=True)
            if counts.max() / len(got) < min_agreement:
                return False
            homes.append(vals[np.argmax(counts)])
        return len(set(homes)) >= 2

    def refine(
        self,
        model,
        min_confidence: float = 0.6,
        min_examples_per_theme: int = 2,
        accept: str = "consistency",
        max_splits: int | None = None,
        results: dict[int, TopicInterpretation] | None = None,
    ) -> list[dict]:
        """
        Interpret all topics and split those the LLM judges ``mixed``.

        A topic is split with ``model.divide`` when the verdict is ``mixed``
        with confidence >= *min_confidence* and at least two sub-themes are
        each backed by >= *min_examples_per_theme* example documents (single
        stray documents do not make a topic mixed).  Whether a split is kept
        is decided by *accept*:

        * ``"consistency"`` (default): the example documents the LLM assigned
          to different sub-themes end up in different new topics (each
          sub-theme's examples mostly in one topic) -- the graph split agrees
          with the LLM's reading;
        * ``"coherence"``: the new topics' size-weighted keyword NPMI beats the
          original topic's;
        * ``"both"`` or ``"none"``.

        Rejected splits are undone.  New topics are interpreted.  Pass
        *results* (from :meth:`interpret`) to reuse existing interpretations.
        Returns a log with one entry per attempted split (``kept`` True/False).
        """
        results = results if results is not None else self.interpret(model)
        candidates = []
        for r in results.values():
            themes = [s for s in r.sub_themes if len(s.get("doc_ids", [])) >= min_examples_per_theme]
            if r.verdict == "mixed" and r.confidence >= min_confidence and len(themes) >= 2:
                candidates.append((r, themes))
        candidates.sort(key=lambda x: -x[0].size)
        if max_splits is not None:
            candidates = candidates[:max_splits]

        log, new_ids = [], []
        verbose = model.config.verbose
        model.config.verbose = False
        try:
            for r, themes in candidates:
                before_labels = model.labels_.copy()
                parent_coh = self._topic_coherence(model, [r.topic_id]).get(r.topic_id, 0.0)
                subs = [s for s in model.divide(r.topic_id, n_subtopics=len(themes)) if s is not None]
                ids = [s.topic_id for s in subs]
                child_coh = self._topic_coherence(model, ids)
                sizes = {s.topic_id: s.size for s in subs}
                total = sum(sizes[t] for t in child_coh) or 1
                weighted = sum(child_coh[t] * sizes[t] for t in child_coh) / total
                consistent = self._split_matches_themes(model.labels_, themes)
                checks = {"consistency": consistent, "coherence": weighted > parent_coh,
                          "both": consistent and weighted > parent_coh, "none": True}
                kept = len(ids) >= 2 and checks[accept]
                if not kept:  # undo the split
                    model.labels_ = before_labels
                    model._extract_topic_info(model.documents_)
                    model._compute_topic_centroids()
                    model._compute_probabilities()
                else:
                    new_ids += ids
                log.append({"topic_id": r.topic_id, "label": r.label, "sub_themes": [s["name"] for s in themes],
                            "new_topics": ids if kept else [], "kept": kept, "coherence_before": parent_coh,
                            "coherence_after": weighted, "consistent": consistent, "evidence": r.evidence})
        finally:
            model.config.verbose = verbose

        # divide()/undo re-extract all topics: restore labels of unchanged topics, interpret the new ones
        for tid, r in results.items():
            t = model.get_topic(tid)
            if t is not None:
                t.label, t.description = r.label, r.description
        if new_ids:
            self.interpret(model, topics=[t for t in new_ids if model.get_topic(t) is not None])
        return log

    def summarize(self, model) -> str:
        """Short overview of the whole topic landscape (uses existing labels/interpretations)."""
        topics = [t for t in model.topics_ if t.topic_id != -1]
        lines = [f"- {t.label or ', '.join(t.keywords[:5])} ({t.size} docs): {t.description or ', '.join(t.keywords[:10])}"
                 for t in topics]
        schema = {"type": "object", "additionalProperties": False, "required": ["overview", "groups"],
                  "properties": {"overview": {"type": "string"},
                                 "groups": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                            "required": ["name", "topics"],
                                            "properties": {"name": {"type": "string"},
                                                           "topics": {"type": "array", "items": {"type": "string"}}}}}}}
        system = (f"You summarise the results of a topic model for a busy reader. Write in {self.language}. "
                  "Give a 3-5 sentence overview of what the corpus is about and how it is distributed, then "
                  "group related topics under broader headings.")
        r = self._complete(system, (f"Corpus: {self.domain_hint}\n" if self.domain_hint else "") + "Topics:\n" + "\n".join(lines),
                           schema, "topic_overview")
        model.overview_ = r
        return r["overview"]
