"""Topic prevalence with confidence intervals, and group comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilson(k, n, z):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def topic_prevalence(
    model,
    groups=None,
    ci: float = 0.95,
    method: str = "wilson",
    n_boot: int = 2000,
    include_outliers: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Share of documents per topic with a confidence interval, optionally per group.

    ``method="wilson"`` uses the Wilson score interval (fast, standard for
    proportions); ``method="bootstrap"`` resamples documents.  With *groups*
    (one value per document, e.g. a metadata column) the shares are computed
    within each group.  Returns ``topic``, ``label``, ``group``, ``n``,
    ``count``, ``share``, ``ci_low``, ``ci_high``.
    """
    from scipy.stats import norm

    labels = np.asarray(model.labels_)
    groups = np.asarray(["all"] * len(labels) if groups is None else groups, dtype=object)
    topics = [t for t in model.topics_ if include_outliers or t.topic_id != -1]
    z = norm.ppf(0.5 + ci / 2)
    rng = np.random.default_rng(random_state)
    rows = []
    for g in pd.unique(groups):
        gl = labels[groups == g]
        if not include_outliers:
            gl = gl[gl != -1]
        n = len(gl)
        boot = None
        if method == "bootstrap" and n:
            boot = gl[rng.integers(0, n, size=(n_boot, n))]
        for t in topics:
            k = int(np.sum(gl == t.topic_id))
            if method == "bootstrap" and boot is not None:
                dist = (boot == t.topic_id).mean(axis=1)
                lo, hi = np.quantile(dist, [(1 - ci) / 2, 1 - (1 - ci) / 2])
            else:
                lo, hi = _wilson(k, n, z)
            rows.append(dict(topic=t.topic_id, label=t.label, group=g, n=n, count=k,
                             share=k / n if n else np.nan, ci_low=float(lo), ci_high=float(hi)))
    return pd.DataFrame(rows)


def compare_groups(model, groups, alpha: float = 0.05) -> pd.DataFrame:
    """
    Which topics are over- or under-represented in which group?

    For every topic, a chi-square test (Fisher's exact test for two groups and
    small counts) of topic membership against group, with Cramér's V as
    effect size and Benjamini-Hochberg-adjusted p-values across topics.
    For two groups also the difference in share (group 2 minus group 1) and
    the odds ratio.  Outliers are excluded.
    """
    from scipy.stats import chi2_contingency, false_discovery_control, fisher_exact

    labels = np.asarray(model.labels_)
    groups = np.asarray(groups, dtype=object)
    keep = labels != -1
    labels, groups = labels[keep], groups[keep]
    levels = list(pd.unique(groups))
    rows = []
    for t in [t for t in model.topics_ if t.topic_id != -1]:
        in_t = labels == t.topic_id
        table = np.array([[np.sum(in_t & (groups == g)), np.sum(~in_t & (groups == g))] for g in levels])
        n = table.sum()
        row = dict(topic=t.topic_id, label=t.label)
        for g, (a, b) in zip(levels, table):
            row[f"share[{g}]"] = a / (a + b) if a + b else np.nan
        chi2, p = chi2_contingency(table, correction=False)[:2]
        if len(levels) == 2 and table.min() < 5:
            p = fisher_exact(table)[1]
        row["cramers_v"] = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else np.nan
        if len(levels) == 2:
            (a1, b1), (a2, b2) = table + 0.5   # Haldane correction
            row["difference"] = row[f"share[{levels[1]}]"] - row[f"share[{levels[0]}]"]
            row["odds_ratio"] = float((a2 / b2) / (a1 / b1))
        row["p_value"] = float(p)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_adjusted"] = false_discovery_control(df.p_value.to_numpy(), method="bh")
    df["significant"] = df.p_adjusted < alpha
    return df.sort_values("p_adjusted").reset_index(drop=True)


def distinctive_keywords(model, groups, group_a, group_b, topic_id: int | None = None, n: int = 10) -> pd.DataFrame:
    """
    Words that distinguish two groups, overall or within one topic.

    Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi &
    Quinn, 2008, "Fightin' Words"), z-scored.  Positive z = typical for
    *group_b*, negative = typical for *group_a*.  Returns the top *n* words
    for each side.
    """
    labels = np.asarray(model.labels_)
    groups = np.asarray(groups, dtype=object)
    kx = model._keyword_extractor
    X = kx.fit_corpus(model.documents_)
    mask = np.ones(len(labels), dtype=bool) if topic_id is None else labels == topic_id
    ya = np.asarray(X[mask & (groups == group_a)].sum(axis=0)).ravel().astype(float)
    yb = np.asarray(X[mask & (groups == group_b)].sum(axis=0)).ravel().astype(float)
    prior = np.asarray(X[mask].sum(axis=0)).ravel().astype(float) + 0.01
    a0 = prior.sum()
    na, nb = ya.sum(), yb.sum()
    delta = (np.log((yb + prior) / (nb + a0 - yb - prior)) - np.log((ya + prior) / (na + a0 - ya - prior)))
    var = 1 / (yb + prior) + 1 / (ya + prior)
    zs = delta / np.sqrt(var)
    vocab = kx._vocabulary
    order = np.argsort(zs)
    rows = [dict(word=vocab[i], z=float(zs[i]), typical_for=group_a) for i in order[:n]]
    rows += [dict(word=vocab[i], z=float(zs[i]), typical_for=group_b) for i in order[::-1][:n]]
    return pd.DataFrame(rows)
