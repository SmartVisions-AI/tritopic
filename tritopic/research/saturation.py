"""Theoretical saturation: would more documents reveal new topics?"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SaturationResult:
    curve: pd.DataFrame            # one row per (fraction, repeat)
    summary: pd.DataFrame          # mean per fraction
    novelty: float                 # share of the final topics not yet found with the second-largest fraction
    late_topics: list              # final topics that only appear with (nearly) all documents
    saturated: bool                # novelty <= 5%
    reference_fraction: float      # the fraction novelty refers to (e.g. 0.8)
    point_95: float | None = None  # smallest share of the data with >= 95% of the topics found
    point_all: float | None = None # smallest share of the data with (almost) all topics found

    def __str__(self) -> str:
        verdict = "saturated" if self.saturated else "not saturated"
        pts = ""
        if self.point_all is not None:
            pts = (f" 95% of the topics are found with {self.point_95:.0%} of the documents, "
                   f"all of them with {self.point_all:.0%}.")
        late = f" {len(self.late_topics)} topic(s) only appear with more data: {self.late_topics}." if self.late_topics else ""
        return (f"Saturation ({verdict}): {1 - self.novelty:.0%} of the final topics are found with "
                f"{self.reference_fraction:.0%} of the documents.{pts}{late}")


def _first(summary: pd.DataFrame, level: float) -> float | None:
    hit = summary[summary.recovered >= level]
    return float(hit.fraction.min()) if len(hit) else None


def saturation_curve(
    model,
    fractions=None,
    method: str = "accumulation",
    min_docs: int | None = None,
    n_repeats: int = 2,
    match_threshold: float = 0.5,
    random_state: int = 0,
) -> SaturationResult:
    """
    Saturation analysis (the "do I have enough data?" question of qualitative research).

    ``method="accumulation"`` (default; exact, instant): imagine reading the
    documents in random order.  A topic becomes discoverable once at least
    *min_docs* (default ``min_cluster_size``) of its documents have been
    seen.  For each fraction of the corpus, the probability that a topic of
    size *s* already reached that threshold is hypergeometric, so the
    expected share of discoverable topics (``recovered``) is exact, without
    refitting.  This is a topic accumulation curve, the analogue of species
    accumulation curves in ecology and of "no new codes in the last
    interviews" in qualitative research.

    ``method="refit"``: refit the model on random subsamples (same settings,
    resolution fixed at ``model.resolution_``) and count how many of the
    final topics are found again (Jaccard >= *match_threshold*).  Empirical
    but slower and noisy, and partly reflects granularity.

    ``novelty`` is the share of the final topics that is not yet
    discoverable with the second-largest fraction (default 80%); a corpus is
    called saturated when novelty <= 5%.  ``late_topics`` lists the topics
    that need (nearly) all documents to appear.
    """
    labels = np.asarray(model.labels_)
    n = len(labels)
    full_topics = [t.topic_id for t in model.topics_ if t.topic_id != -1]
    if method == "accumulation":
        from scipy.stats import hypergeom

        fr = np.asarray(fractions if fractions is not None else np.round(np.arange(0.05, 1.0001, 0.05), 2), float)
        m = min_docs or model.config.min_cluster_size
        sizes = {t: int(np.sum(labels == t)) for t in full_topics}
        rows, probs = [], {}
        for f in fr:
            k = int(round(f * n))
            p = {t: float(hypergeom.sf(m - 1, n, s, k)) for t, s in sizes.items()}
            probs[f] = p
            rows.append(dict(fraction=float(f), repeat=0, n_docs=k, n_topics=sum(p.values()),
                             recovered=sum(p.values()) / max(len(full_topics), 1)))
        curve = pd.DataFrame(rows)
        summary = curve[["fraction", "n_docs", "n_topics", "recovered"]].copy()
        ref = float(fr[fr < 1.0].max()) if np.any(fr < 1.0) else 1.0
        ref = float(fr[np.argmin(np.abs(fr - 0.8))]) if np.any(np.isclose(fr, 0.8)) else ref
        late = [t for t, pt in probs[ref].items() if pt < 0.5]
        novelty = float(1 - summary.loc[np.isclose(summary.fraction, ref), "recovered"].iloc[0])
        return SaturationResult(curve=curve, summary=summary, novelty=novelty, late_topics=late,
                                saturated=novelty <= 0.05, reference_fraction=ref,
                                point_95=_first(summary, 0.95), point_all=_first(summary, 0.999))
    if method != "refit":
        raise ValueError("method must be 'accumulation' or 'refit'")
    fractions = fractions if fractions is not None else (0.2, 0.4, 0.6, 0.8, 1.0)

    from tritopic.core.model import TriTopic
    from tritopic.research.reliability import _best_match_jaccard

    rng = np.random.default_rng(random_state)
    emb = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    rows = []
    for frac in fractions:
        for rep in range(n_repeats if frac < 1.0 else 1):
            idx = np.arange(n) if frac >= 1.0 else np.sort(rng.choice(n, int(frac * n), replace=False))
            cfg = copy.deepcopy(model.config)
            cfg.verbose = False
            cfg.auto_resolution = False
            cfg.resolution = float(model.resolution_)
            sub = TriTopic(config=cfg)
            sub.fit([model.documents_[i] for i in idx], embeddings=emb[idx])
            sub_labels = -np.ones(n, dtype=int)
            sub_labels[idx] = sub.labels_
            in_sample = np.zeros(n, dtype=bool)
            in_sample[idx] = True
            recovered = []
            for t in full_topics:
                members = np.where((labels == t) & in_sample)[0]
                if len(members) and _best_match_jaccard(idx, sub_labels, members)[0] >= match_threshold:
                    recovered.append(t)
            rows.append(dict(fraction=frac, repeat=rep, n_docs=len(idx),
                             n_topics=len(set(sub.labels_.tolist()) - {-1}),
                             recovered=len(recovered) / max(len(full_topics), 1), recovered_ids=recovered))
    curve = pd.DataFrame(rows)
    summary = curve.groupby("fraction")[["n_docs", "n_topics", "recovered"]].mean().reset_index()
    ref = summary.fraction[summary.fraction < 1.0].max()
    ref_rows = curve[curve.fraction == ref]
    late = []
    for t in full_topics:
        found = 0
        for _, r in ref_rows.iterrows():
            found += t in r["recovered_ids"]
        if found < len(ref_rows) / 2:
            late.append(t)
    novelty = float(1 - summary.loc[summary.fraction == ref, "recovered"].iloc[0]) if len(ref_rows) else 0.0
    return SaturationResult(curve=curve.drop(columns="recovered_ids"), summary=summary, novelty=novelty,
                            late_topics=late, saturated=novelty <= 0.05, reference_fraction=float(ref),
                            point_95=_first(summary, 0.95), point_all=_first(summary, 0.999))


def plot_saturation(result: SaturationResult):
    """Plotly figure: share of the final topics already found, against corpus size."""
    import plotly.graph_objects as go

    s = result.summary
    fig = go.Figure()
    fig.add_scatter(x=s.n_docs, y=s.recovered, mode="markers+lines", name="final topics found")
    fig.add_hline(y=0.95, line_dash="dot", annotation_text="95%")
    fig.update_layout(title="Topic saturation", xaxis_title="documents", yaxis_title="share of final topics found",
                      yaxis_tickformat=".0%")
    return fig
