"""
Research charts for a fitted TriTopic model (Plotly).

Every function returns a ``plotly.graph_objects.Figure``; call ``.show()`` or
``.write_html(path)``.  Topics are coloured by size rank with a fixed
eight-colour palette; further topics are grey.

>>> from tritopic.visualization import charts
>>> charts.plot_triview(model).show()
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#9AA5B1"
INK, MUTED, LINE, PETROL, AMBER, GOOD = "#0F2239", "#65758B", "#DCE3EB", "#2E86AB", "#F59E0B", "#0B7A3B"
FONT = dict(family="Lato, Calibri, Helvetica, Arial, sans-serif", color=INK, size=13)


def _ids(model) -> list[int]:
    return [t.topic_id for t in sorted(model.topics_, key=lambda t: -t.size) if t.topic_id != -1]


def topic_colors(model) -> dict[int, str]:
    """Topic id -> colour (by size rank; topics beyond eight are grey)."""
    return {t: PALETTE[i] if i < len(PALETTE) else OTHER for i, t in enumerate(_ids(model))}


def _name(model, t: int, n: int = 30) -> str:
    info = model.get_topic(t)
    name = info.label or ", ".join(info.keywords[:3])
    return name if len(name) <= n else name[: n - 1].rstrip(" ,") + "…"


def _layout(fig, title: str, height: int = 520, **kw):
    kw.setdefault("margin", dict(l=20, r=20, t=60, b=40))
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=18)), font=FONT, height=height,
                      plot_bgcolor="white", paper_bgcolor="white", **kw)
    return fig


def _wrap(text: str, width: int) -> str:
    return "<br>".join(textwrap.wrap(str(text), width))


def _cards(cards: list[dict], title: str, cols: int = 3, card_h: float = 1.0, height: int | None = None):
    """Grid of text cards drawn with shapes and annotations (no external HTML)."""
    fig = go.Figure()
    rows = int(np.ceil(len(cards) / cols)) or 1
    for i, c in enumerate(cards):
        r, k = divmod(i, cols)
        x0, y0 = k + 0.03, -(r * card_h) - 0.03
        fig.add_shape(type="rect", x0=x0, x1=x0 + 0.94, y0=y0, y1=y0 - card_h + 0.06, line=dict(color=c.get("border", LINE), width=2),
                      fillcolor=c.get("fill", "white"), layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x0 + 0.94, y0=y0, y1=y0 - 0.025, line=dict(width=0), fillcolor=c.get("color", LINE))
        ty = y0 - 0.06
        if c.get("big"):
            fig.add_annotation(x=x0 + 0.04, y=ty, text=f"<b>{c['big']}</b>", showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=30, color=INK, family="Playfair Display, Georgia, serif"))
            ty -= 0.2 * card_h
        fig.add_annotation(x=x0 + 0.04, y=ty, text=c["text"], showarrow=False, xanchor="left", yanchor="top",
                           align="left", font=dict(size=12, color=INK))
    fig.update_xaxes(visible=False, range=[0, cols])
    fig.update_yaxes(visible=False, range=[-rows * card_h, 0])
    return _layout(fig, title, height=height or int(rows * 260 + 80))


# ============================================================================================
def plot_topic_table(model, reliability: pd.DataFrame | None = None, cols: int = 4):
    """
    The 'periodic table' of topics: one tile per topic with a two-letter
    symbol, size, share of the corpus, reliability (pass the result of
    :func:`tritopic.research.topic_reliability`) and keyword coherence (NPMI).
    """
    info = model.get_topic_info().set_index("Topic")
    if "Coherence" not in info or info["Coherence"].isna().all():
        model.evaluate()
        info = model.get_topic_info().set_index("Topic")
    rel = reliability.set_index("topic")["reliability"].to_dict() if reliability is not None else {}
    colors, used, n = topic_colors(model), set(), len(model.labels_)
    cards = []
    for i, t in enumerate(_ids(model)):
        topic = model.get_topic(t)
        words = [w for w in (topic.label or " ".join(topic.keywords[:2])).replace("-", " ").split() if len(w) > 2
                 and w.lower() not in {"and", "the", "news", "era"}] or ["Topic"]
        cands = [words[0][0] + (words[1] if len(words) > 1 else words[0])[0]] + [words[0][0] + ch for ch in words[0][1:]]
        sym = next((s[0].upper() + s[1].lower() for s in cands if (s[0].upper() + s[1].lower()) not in used), words[0][:2])
        used.add(sym)
        coh = info.loc[t, "Coherence"] if t in info.index and "Coherence" in info else np.nan
        cards.append(dict(color=colors[t], fill="white", big=sym, text=(
            f"<b>{_wrap(_name(model, t, 40), 26)}</b><br><span style='color:{MUTED}'>#{i + 1} · {topic.size} docs</span><br>"
            f"<span style='color:{MUTED}'>{', '.join(topic.keywords[:3])}</span><br><br>"
            f"share <b>{topic.size / n:.0%}</b>   reliab. <b>{rel.get(t, float('nan')):.2f}</b>   NPMI <b>{coh:.2f}</b>")))
    return _cards(cards, "Topics at a glance", cols=cols, card_h=1.0, height=int(np.ceil(len(cards) / cols) * 240 + 80))


# ============================================================================================
def plot_triview(model, zoom: float = 0.3, show_documents: bool = True):
    """
    Tri-View triangle: for the links inside each topic, the share that exists
    only in the meaning view, only in the wording view, or in both
    (:func:`tritopic.research.view_composition`).  The triangle is zoomed to
    the region with at least *zoom* meaning-only links; set ``zoom=0`` for the
    full triangle.
    """
    from tritopic.research.structure import view_composition

    tops, docs = view_composition(model)
    colors = topic_colors(model)
    fig = go.Figure()
    if show_documents and len(docs):
        d = docs[(docs.sem_only + docs.lex_only + docs.both) >= 3].copy()
        tot = d.sem_only + d.lex_only + d.both
        fig.add_scatterternary(a=d.both / tot, b=d.sem_only / tot, c=d.lex_only / tot, mode="markers", showlegend=False,
                               marker=dict(size=4, color=[colors.get(t, OTHER) for t in d.topic], opacity=0.3),
                               hoverinfo="skip")
    for r in tops.itertuples():
        fig.add_scatterternary(a=[r.both_share], b=[r.sem_share], c=[r.lex_share], mode="markers", name=_name(model, r.topic),
                               marker=dict(size=10 + np.sqrt(model.get_topic(r.topic).size), color=colors.get(r.topic, OTHER),
                                           line=dict(color="white", width=2)),
                               hovertemplate=f"<b>{_name(model, r.topic, 60)}</b><br>meaning only %{{b:.0%}}<br>wording only %{{c:.0%}}<br>both %{{a:.0%}}<extra></extra>")
    fig.update_layout(ternary=dict(sum=1, aaxis=dict(title="both views", min=0, tickformat=".0%"),
                                   baxis=dict(title="meaning only", min=zoom, tickformat=".0%"),
                                   caxis=dict(title="wording only", min=0, tickformat=".0%"), bgcolor="white"))
    return _layout(fig, "What holds each topic together?", height=620)


# ============================================================================================
def plot_resolution_ladder(model, multipliers=(0.25, 0.5, 1, 3, 9), ladder=None):
    """
    Zoom ladder: Sankey of the clusters at several resolutions
    (:func:`tritopic.research.resolution_ladder`).  Colour = the model topic
    most of a cluster's documents belong to.
    """
    from tritopic.research.structure import resolution_ladder

    lad = ladder or resolution_ladder(model, multipliers=multipliers)
    colors = topic_colors(model)
    L = lad.levels.reset_index(drop=True)
    key = {(r.level, r.cluster): i for i, r in enumerate(L.itertuples())}
    n_levels = L.level.max() + 1
    node = dict(label=[f"{r.keywords} ({r.size})" for r in L.itertuples()],
                color=[colors.get(r.main_topic, OTHER) for r in L.itertuples()],
                x=[0.01 + 0.98 * r.level / max(n_levels - 1, 1) for r in L.itertuples()], pad=8, thickness=14)
    F = lad.flows
    link = dict(source=[key[(r.level, r.source)] for r in F.itertuples()], target=[key[(r.level + 1, r.target)] for r in F.itertuples()],
                value=list(F["count"]), color=["rgba(46,134,171,0.18)"] * len(F))
    fig = go.Figure(go.Sankey(node=node, link=link, arrangement="snap"))
    for lv in range(n_levels):
        sub = L[L.level == lv]
        fig.add_annotation(x=lv / max(n_levels - 1, 1), y=1.08, xref="paper", yref="paper", showarrow=False,
                           text=f"<b>{len(sub)} topics</b><br>×{sub.multiplier.iloc[0]}")
    return _layout(fig, "How the topics split when you zoom in", height=620, margin=dict(l=20, r=20, t=110, b=20))


# ============================================================================================
def plot_topic_onions(model, cols: int = 4):
    """
    Topic onions: per topic, documents in rings from core (centre) to edge,
    coloured by stability (run :func:`tritopic.research.topic_reliability`
    first), bridge documents ringed (:func:`tritopic.research.topic_cores`).
    """
    from tritopic.research.structure import topic_cores

    C = topic_cores(model)
    ids, colors = _ids(model), topic_colors(model)
    rows = int(np.ceil(len(ids) / cols))
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "polar"}] * cols for _ in range(rows)],
                        subplot_titles=[_name(model, t, 28) for t in ids], horizontal_spacing=0.06, vertical_spacing=0.12)
    has_stab = C.stability.notna().any()
    for i, t in enumerate(ids):
        d = C[C.topic == t]
        r, k = divmod(i, cols)
        fig.add_scatterpolar(r=np.sqrt(d.core_rank), theta=np.degrees(d.angle), mode="markers", showlegend=False, row=r + 1, col=k + 1,
                             marker=dict(size=5, color=d.stability if has_stab else PETROL, cmin=0.5, cmax=1,
                                         colorscale=[[0, AMBER], [1, PETROL]], showscale=bool(has_stab and i == 0),
                                         colorbar=dict(title="stability", tickformat=".0%", len=0.5) if has_stab else None,
                                         line=dict(color=[colors.get(b, OTHER) if s >= 0.4 and b != -1 else "rgba(0,0,0,0)"
                                                          for b, s in zip(d.bridge_to, d.bridge_share)], width=2)),
                             text=[model.documents_[j][:90] for j in d.doc], hovertemplate="%{text}<extra></extra>")
    fig.update_polars(radialaxis=dict(range=[0, 1], tickvals=[0.5, 0.707, 0.866], showticklabels=False),
                      angularaxis=dict(showticklabels=False))
    return _layout(fig, "Core and edge of every topic", height=rows * 300 + 80)


# ============================================================================================
def plot_keyword_barcode(model, topic_id: int, n_keywords: int = 10):
    """
    Keyword barcode: which documents of a topic (core → edge) contain each
    keyword; coverage inside vs. outside the topic in the labels
    (:func:`tritopic.research.keyword_coverage`).
    """
    from tritopic.research.structure import keyword_coverage

    K = keyword_coverage(model, topic_id=topic_id, n_keywords=n_keywords)
    z = np.array(K.presence.tolist())
    y = [f"{r.keyword}  ({r.coverage:.0%} in · {r.outside:.0%} out)" for r in K.itertuples()]
    fig = go.Figure(go.Heatmap(z=z, y=y, colorscale=[[0, "#E8EFF5"], [1, "#1A3A5C"]], showscale=False, xgap=0, ygap=3,
                               hovertemplate="document %{x}<br>%{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="documents of the topic, core → edge", showticklabels=False)
    return _layout(fig, f"Keyword coverage: {_name(model, topic_id, 50)}", height=60 + 34 * len(K) + 60,
                   margin=dict(l=260, r=20, t=60, b=50))


# ============================================================================================
def plot_constellation(model, top_links: int = 12):
    """
    Constellation: topics placed by centroid similarity (MDS), circle area by
    size, lines by actual cross-topic neighbour links with the number of
    bridge documents (:func:`tritopic.research.topic_connections`).
    """
    from sklearn.manifold import MDS
    from tritopic.research.bridges import topic_connections

    ids, colors = _ids(model), topic_colors(model)
    E = model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    labels = np.asarray(model.labels_)
    C = np.array([E[labels == t].mean(0) for t in ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    pos = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4).fit_transform(np.clip(1 - C @ C.T, 0, None))
    P = dict(zip(ids, pos))
    conn = topic_connections(model)
    conn = conn[conn.strength > 0].head(top_links)
    fig = go.Figure()
    smax = conn.strength.max() if len(conn) else 1
    for r in conn.itertuples():
        a, b = P[r.topic_a], P[r.topic_b]
        fig.add_scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color=MUTED, width=1 + 9 * r.strength / smax), opacity=0.3 + 0.5 * r.strength / smax)
        if r.bridge_docs:
            fig.add_annotation(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2, text=str(r.bridge_docs), showarrow=False,
                               bgcolor="white", bordercolor=LINE, font=dict(size=11))
    fig.add_scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[_name(model, t, 26) for t in ids], textposition="bottom center",
                    marker=dict(size=[12 + np.sqrt(model.get_topic(t).size) * 1.6 for t in ids], color=[colors[t] for t in ids],
                                line=dict(color="white", width=3)), showlegend=False,
                    hovertemplate="%{text}<extra></extra>")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _layout(fig, "Which topics are neighbours?", height=560)


# ============================================================================================
def plot_coassignment(model=None, result=None, **kwargs):
    """
    Fuzzy borders: share of refits in which two documents share a topic
    (:func:`tritopic.research.coassignment`; pass ``result`` to reuse one).
    """
    from tritopic.research.structure import coassignment

    res = result or coassignment(model, **kwargs)
    fig = go.Figure(go.Heatmap(z=res.matrix, colorscale=[[0, "white"], [1, PETROL]], zmin=0, zmax=1,
                               colorbar=dict(title="same topic", tickformat=".0%"),
                               hovertemplate="same topic in %{z:.0%} of the refits<extra></extra>"))
    t = res.topics
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] != t[start]:
            fig.add_shape(type="rect", x0=start - 0.5, x1=i - 0.5, y0=start - 0.5, y1=i - 0.5, line=dict(color=INK, width=1))
            if model is not None:
                fig.add_annotation(x=-1, y=(start + i) / 2, text=_name(model, int(t[start]), 22), showarrow=False, xanchor="right", font=dict(size=11))
            start = i
    fig.update_yaxes(autorange="reversed", showticklabels=False, scaleanchor="x")
    fig.update_xaxes(showticklabels=False)
    return _layout(fig, f"How sharp are the topic borders? ({res.n_refits} refits)", height=640, margin=dict(l=170, r=20, t=60, b=20))


# ============================================================================================
def plot_group_tilt(model, groups):
    """
    Group tilt: share of each topic per group with 95% confidence intervals
    (:func:`tritopic.research.topic_prevalence`) and, per topic, Cramér's V
    and the Benjamini-Hochberg significance (:func:`tritopic.research.compare_groups`).
    """
    from tritopic.research.prevalence import compare_groups, topic_prevalence

    prev = topic_prevalence(model, groups=groups)
    tests = compare_groups(model, groups).set_index("topic")
    ids = _ids(model)
    names = [_name(model, t, 32) for t in ids]
    fig = go.Figure()
    gcol = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
    for gi, (g, d) in enumerate(prev.groupby("group", sort=False)):
        d = d.set_index("topic").loc[ids]
        fig.add_scatter(x=d.share, y=names, mode="markers", name=str(g), marker=dict(size=11, color=gcol[gi % len(gcol)]),
                        error_x=dict(type="data", symmetric=False, array=d.ci_high - d.share, arrayminus=d.share - d.ci_low,
                                     thickness=1.5, width=0), offsetgroup=str(g))
    for t, nm in zip(ids, names):
        if t in tests.index:
            r = tests.loc[t]
            fig.add_annotation(x=1.0, xref="paper", y=nm, text=f"{'* ' if r.significant else ''}V = {r.cramers_v:.2f}",
                               showarrow=False, xanchor="left", font=dict(size=11, color=MUTED))
    fig.update_layout(scattermode="group")
    fig.update_xaxes(tickformat=".0%", gridcolor=LINE, title="share of the group's documents (95% CI)")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Do the groups cover different topics?", height=120 + 46 * len(ids), margin=dict(l=20, r=110, t=60, b=50))


# ============================================================================================
def plot_trust(model, reliability: pd.DataFrame):
    """Trust quadrant: topic size (log) against reliability, with the 0.7 and 0.5 guides."""
    colors = topic_colors(model)
    R = reliability.copy()
    fig = go.Figure()
    fig.add_hline(y=0.7, line=dict(color=GOOD, dash="dash"), annotation_text="report with confidence", annotation_position="top left")
    fig.add_hline(y=0.5, line=dict(color=AMBER, dash="dash"), annotation_text="check before reporting", annotation_position="bottom left")
    fig.add_scatter(x=R["size"], y=R.reliability, mode="markers+text", text=[_name(model, t, 26) for t in R.topic], textposition="middle right",
                    marker=dict(size=14, color=[colors.get(t, OTHER) for t in R.topic], line=dict(color="white", width=2)),
                    customdata=R.core_share, hovertemplate="%{text}<br>%{x} documents · reliability %{y:.2f} · core %{customdata:.0%}<extra></extra>",
                    showlegend=False)
    fig.update_xaxes(type="log", title="documents in the topic (log scale)", gridcolor=LINE)
    fig.update_yaxes(range=[0, 1.05], title="reliability", gridcolor=LINE)
    return _layout(fig, "Which topics can go into the paper?", height=480)


# ============================================================================================
def plot_composition(model, reference, reference_name: str = "class"):
    """Composition: share of each reference category (e.g. known classes) inside every topic."""
    ref = np.asarray(reference)
    labels = np.asarray(model.labels_)
    ids = _ids(model)
    cats = list(pd.Series(ref).value_counts().index)
    fig = go.Figure()
    for ci, c in enumerate(cats):
        share = [np.mean(ref[labels == t] == c) for t in ids]
        fig.add_bar(y=[f"{_name(model, t, 30)} ({model.get_topic(t).size})" for t in ids], x=share, name=str(c), orientation="h",
                    marker_color=PALETTE[ci % len(PALETTE)], text=[f"{s:.0%}" if s >= 0.12 else "" for s in share], textposition="inside")
    fig.update_layout(barmode="stack", legend_title=reference_name)
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "What is inside each topic?", height=120 + 44 * len(ids))


# ============================================================================================
def plot_birth_timeline(evolution, highlight: dict | None = None):
    """
    Birth timeline of a :func:`tritopic.research.topic_evolution` result:
    new topics above the axis, ending topics below, circle area = documents.
    *highlight* maps a name to a keyword regex, e.g. ``{"COVID": "covid|coronavirus"}``.
    """
    import re

    E = evolution.events
    periods = list(dict.fromkeys(evolution.nodes.period))
    x = {p: i for i, p in enumerate(periods)}
    fig = go.Figure()
    for kind, sign, filled in [("birth", 1, True), ("death", -1, False)]:
        d = E[E.event == kind]
        xs, ys, sizes, texts, cols = [], [], [], [], []
        for p, g in d.groupby("period", sort=False):
            off = 0.0
            for r in g.sort_values("size", ascending=False).itertuples():
                rad = 0.12 + np.sqrt(r.size) * 0.018
                off += rad
                xs.append(x[p]); ys.append(sign * (0.25 + off)); off += rad + 0.04
                sizes.append(8 + np.sqrt(r.size) * 2.2)
                hit = next((h for h, rx in (highlight or {}).items() if re.search(rx, r.keywords)), None)
                texts.append(f"<b>{p} · {'born' if kind == 'birth' else 'ends'}{' · ' + hit if hit else ''}</b><br>{r.size} documents<br>{r.keywords}")
                cols.append(AMBER if hit else PETROL)
        fig.add_scatter(x=xs, y=ys, mode="markers", name="new topic" if kind == "birth" else "topic ends", hovertext=texts, hoverinfo="text",
                        marker=dict(size=sizes, color=cols if filled else "white", line=dict(color=cols if filled else MUTED, width=1.5)))
    carry = E[E.event.isin(["continuation", "split", "merge"])].groupby("period").size()
    fig.add_scatter(x=list(range(len(periods))), y=[0] * len(periods), mode="text", showlegend=False, hoverinfo="skip",
                    text=[f"<b>{p}</b><br>{carry.get(p, 0)} carry on" for p in periods])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, zeroline=True)
    return _layout(fig, "When did themes appear, and when did they fade?", height=560)


# ============================================================================================
def plot_codebook_coverage(models: dict):
    """
    Codebook coverage: one bar per seeded model (``{"name": model}``), split
    into topics; solid = grew from a seed, hatched = emerged on its own.
    """
    from tritopic.research.structure import codebook_coverage

    fig = go.Figure()
    seed_names: list[str] = []
    for m in models.values():
        for t in m.topics_:
            if t.seed and t.seed not in seed_names:
                seed_names.append(t.seed)
    scol = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seed_names)}
    for name, m in models.items():
        cov = codebook_coverage(m)
        for r in cov.itertuples():
            seed = r.seed if isinstance(r.seed, str) else None
            fig.add_bar(y=[name], x=[r.share], orientation="h", showlegend=False,
                        marker=dict(color=scol.get(seed, "#FCE3B6"), pattern_shape="" if seed else "/", line=dict(color="white", width=2)),
                        text=(seed or "emerged: " + r.keywords.split(", ")[0]) if r.share > 0.07 else "", textposition="inside",
                        hovertemplate=f"{seed or 'emerged'}<br>{r.size} documents ({r.share:.0%})<br>{r.keywords}<extra></extra>")
        seeded = cov[cov.seed.map(lambda v: isinstance(v, str))].share.sum()
        fig.add_annotation(x=1, y=name, xref="paper", text=f"seeded {seeded:.0%} · emerged {1 - seeded:.0%}", showarrow=False,
                           xanchor="left", font=dict(size=11, color=MUTED))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "How much does the codebook explain?", height=140 + 90 * len(models), margin=dict(l=20, r=170, t=60, b=40))


# ============================================================================================
def plot_quote_wall(model, quotes: pd.DataFrame | None = None, cols: int = 3):
    """Quote wall: the most telling sentence of every topic (:func:`tritopic.research.topic_quotes`)."""
    from tritopic.research.quotes import topic_quotes

    q = quotes if quotes is not None else topic_quotes(model, n=1)
    colors = topic_colors(model)
    cards = []
    for t in _ids(model):
        qt = q[(q.topic == t)].sort_values("rank")
        if not len(qt):
            continue
        cards.append(dict(color=colors[t], text=f"<i style='font-size:14px'>“{_wrap(qt.quote.iloc[0], 46)}”</i><br><br>"
                                                f"<span style='color:{MUTED}'>{_name(model, t, 44)} · {model.get_topic(t).size} docs</span>"))
    return _cards(cards, "One sentence per topic", cols=cols)


# ============================================================================================
def plot_verdict_board(results: dict, refine_log: list | None = None):
    """
    LLM verdict board: the topics of a :meth:`TopicInterpreter.interpret`
    result in three columns (coherent, mixed, unclear) with confidence,
    evidence, sub-themes and, if given, the outcome of ``refine()``.
    """
    log = {e["topic_id"]: e for e in (refine_log or [])}
    fig = go.Figure()
    cols = ["coherent", "mixed", "unclear"]
    height_units = 0
    for k, verdict in enumerate(cols):
        items = sorted([r for r in results.values() if r.verdict == verdict], key=lambda r: -r.size)
        fig.add_annotation(x=k + 0.03, y=0, text=f"<b>{verdict.capitalize()}</b> ({len(items)})", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=15))
        y = -0.05
        for r in items:
            lines = [f"<b>{_wrap(r.label, 34)}</b>", f"<span style='color:{MUTED}'>{r.size} docs · confidence {r.confidence:.0%}</span>"]
            if r.sub_themes:
                lines.append("sub-themes: " + _wrap(" | ".join(s["name"] for s in r.sub_themes), 40))
            lines.append(f"<span style='color:{MUTED}'>{_wrap(r.evidence, 44)}</span>")
            e = log.get(r.topic_id)
            if e:
                lines.append(f"<b style='color:{GOOD}'>refine(): {'split kept' if e['kept'] else 'split undone'}</b>")
            text = "<br>".join(lines)
            h = 0.06 * (text.count("<br>") + 2)
            fig.add_shape(type="rect", x0=k + 0.03, x1=k + 0.97, y0=y, y1=y - h, line=dict(color=AMBER if verdict == "mixed" else LINE, width=2),
                          fillcolor="white", layer="below")
            fig.add_annotation(x=k + 0.06, y=y - 0.02, text=text, showarrow=False, xanchor="left", yanchor="top", align="left", font=dict(size=11))
            y -= h + 0.05
        height_units = max(height_units, -y)
    fig.update_xaxes(visible=False, range=[0, 3])
    fig.update_yaxes(visible=False, range=[-max(height_units, 0.5), 0.12])
    return _layout(fig, "What the LLM thought of each topic", height=int(150 + 380 * max(height_units, 0.5)))


# ============================================================================================
def plot_coder_confusion(result, model):
    """
    Coder confusion grid of an :func:`intercoder_reliability` result: rows =
    TriTopic's topic, columns = the LLM coder's choice; diagonal in petrol,
    disagreements in amber.
    """
    C = result.confusion
    ids = [t for t in _ids(model) if t in C.index]
    M = C.loc[ids, ids].values
    names = [_name(model, t, 24) for t in ids]
    diag = np.where(np.eye(len(ids), dtype=bool), M, np.nan)
    off = np.where(~np.eye(len(ids), dtype=bool) & (M > 0), M, np.nan)
    fig = go.Figure()
    fig.add_heatmap(z=diag, x=names, y=names, colorscale=[[0, "#D5E7F0"], [1, PETROL]], showscale=False, xgap=2, ygap=2,
                    hovertemplate="TriTopic %{y}<br>LLM %{x}<br>%{z} documents<extra></extra>")
    fig.add_heatmap(z=off, x=names, y=names, colorscale=[[0, "#FCE3B6"], [1, AMBER]], showscale=False, xgap=2, ygap=2,
                    hovertemplate="TriTopic %{y}<br>LLM %{x}<br>%{z} documents<extra></extra>")
    for i in range(len(ids)):
        for j in range(len(ids)):
            if M[i, j]:
                fig.add_annotation(x=names[j], y=names[i], text=str(M[i, j]), showarrow=False, font=dict(size=12))
    fig.update_yaxes(autorange="reversed", title="TriTopic")
    fig.update_xaxes(side="top", tickangle=-35, title="LLM coder")
    return _layout(fig, f"Where does the second coder disagree?  κ = {result.kappa:.2f}", height=160 + 48 * len(ids),
                   margin=dict(l=20, r=20, t=170, b=20))


# ============================================================================================
def plot_topic_discovery(model, min_docs: int | None = None):
    """
    Saturation with topic dots: expected share of visible topics while
    reading the corpus in random order, and per topic when it is visible with
    50% (dot) and 95% (bar end) probability (:func:`tritopic.research.topic_discovery`).
    """
    from tritopic.research.structure import topic_discovery

    T, curve = topic_discovery(model, min_docs=min_docs)
    colors = topic_colors(model)
    xmax = min(1.0, np.ceil(np.nanmax(T.f95) * 20 + 1) / 20) if len(T) else 1.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.06)
    c = curve[curve.fraction <= xmax]
    fig.add_scatter(x=c.fraction, y=c.recovered, mode="lines", line=dict(color=PETROL, width=3), name="topics visible", row=1, col=1)
    names = [_name(model, t, 28) for t in T.topic]
    for r, nm in zip(T.itertuples(), names):
        col = colors.get(r.topic, OTHER)
        fig.add_scatter(x=[r.f50, r.f95], y=[nm, nm], mode="lines", line=dict(color=col, width=7), opacity=0.45, showlegend=False,
                        hoverinfo="skip", row=2, col=1)
        fig.add_scatter(x=[r.f50], y=[nm], mode="markers", marker=dict(size=12, color=col), showlegend=False, row=2, col=1,
                        hovertemplate=f"{nm}: 50% likely visible after {r.f50:.0%}, 95% after {r.f95:.0%}<extra></extra>")
        fig.add_scatter(x=[r.f50], y=[np.interp(r.f50, curve.fraction, curve.recovered)], mode="markers", marker=dict(size=9, color=col),
                        showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title="topics visible", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(tickformat=".0%", range=[0, xmax], gridcolor=LINE)
    fig.update_xaxes(title="share of the corpus read", row=2, col=1)
    return _layout(fig, "When does each topic become visible?", height=300 + 34 * len(T), showlegend=False)
