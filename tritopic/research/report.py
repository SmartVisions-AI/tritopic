"""A methods paragraph and parameter table for papers."""

from __future__ import annotations

import platform
from importlib import metadata as _md

import numpy as np


def _version(pkg: str) -> str:
    try:
        return _md.version(pkg)
    except Exception:
        return "n/a"


def methods_report(model, corpus: str = "the corpus", fmt: str = "markdown") -> str:
    """
    Write the methods section for a TriTopic analysis: what was done, with
    which settings, how the number of topics was chosen, quality metrics,
    and software versions.  Uses only the fitted model (no LLM), so the text
    is reproducible.  ``fmt="markdown"`` adds a parameter table; ``"text"``
    returns the paragraph only.
    """
    import tritopic

    c = model.config
    n = len(model.documents_)
    words = int(np.mean([len(d.split()) for d in model.documents_]))
    topics = [t for t in model.topics_ if t.topic_id != -1]
    outliers = float(np.mean(np.asarray(model.labels_) == -1))
    m = model.evaluate() if topics else {}
    views = ["semantic (sentence embeddings)"]
    if c.use_lexical_view:
        views.append("lexical (TF-IDF)")
    if c.use_metadata_view and getattr(model, "_metadata_graph", None) is not None:
        views.append("metadata")

    if getattr(model, "seeds_", None):
        k_text = (f"The analysis was seeded with {len(model.seeds_)} predefined topics "
                  f"({', '.join(model.seeds_)}); the documents most similar to each seed were fixed to its topic, "
                  f"all other documents could join a seeded topic or form new ones. "
                  f"{len(model.emergent_topics_)} additional topics emerged from the data.")
    elif model.n_topics == "auto" and c.auto_resolution and model.resolution_search_:
        k_text = (f"The number of topics was chosen automatically: Leiden was run at "
                  f"{len(model.resolution_search_)} resolutions, and the coarsest resolution whose mean keyword "
                  f"coherence (NPMI) was within {c.auto_resolution_tolerance:.0%} of the best was kept "
                  f"(resolution {model.resolution_:.3f}).")
    elif isinstance(model.n_topics, int):
        k_text = f"The number of topics was set to {model.n_topics}; the Leiden resolution was searched accordingly."
    else:
        k_text = f"The Leiden resolution was fixed at {model.resolution_:.3f}."

    para = (
        f"Topics were identified with TriTopic {tritopic.__version__} (Egger, 2026). {corpus[0].upper() + corpus[1:]} "
        f"comprised {n:,} documents (mean length {words} words). Documents were embedded with "
        f"{c.embedding_model}"
        + (f" and reduced to {c.reduced_dims} dimensions with {c.dim_reduction_method.upper()}" if c.use_dim_reduction else "")
        + f". A document graph was built from {', '.join(views)} views ({c.n_neighbors} nearest neighbours, "
        f"{c.graph_type} graph) and partitioned with consensus Leiden clustering over {c.n_consensus_runs} runs"
        + (" with iterative refinement" if c.use_iterative_refinement else "")
        + f". {k_text} Clusters smaller than {c.min_cluster_size} documents were treated as outliers. "
        f"Topics were described by {c.n_keywords} keywords from coverage-weighted class-based TF-IDF "
        f"({c.keyword_method}). The final model has {len(topics)} topics"
        + (f" ({outliers:.1%} outlier documents)" if outliers else " (all documents assigned)")
        + (f"; mean keyword coherence (NPMI) {m['coherence_mean']:.3f}, topic diversity {m['diversity']:.2f}, "
           f"consensus stability (mean pairwise ARI) {m['stability']:.3f}" if m else "")
        + f". Random seed: {c.random_state}."
    )
    if getattr(model, "interpretations_", None):
        para += (" Topic labels and descriptions were produced with an LLM (TopicInterpreter), which read each "
                 "topic's keywords, proportionally sampled example documents and neighbouring topics.")
    if fmt == "text":
        return para
    rows = [("TriTopic", tritopic.__version__), ("Python", platform.python_version()),
            ("sentence-transformers", _version("sentence-transformers")), ("leidenalg", _version("leidenalg")),
            ("umap-learn", _version("umap-learn")), ("Documents", f"{n:,}"), ("Embedding model", c.embedding_model),
            ("Reduced dimensions", c.reduced_dims if c.use_dim_reduction else "none"),
            ("Graph", f"{c.graph_type}, k = {c.n_neighbors}"), ("Views", ", ".join(views)),
            ("Consensus runs", c.n_consensus_runs), ("Resolution", f"{model.resolution_:.3f}"),
            ("Minimum topic size", c.min_cluster_size), ("Topics", len(topics)), ("Outliers", f"{outliers:.1%}"),
            ("Random seed", c.random_state)]
    table = "| Setting | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    return f"{para}\n\n{table}\n\n*Reference:* Egger, R. (2026). TriTopic: Tri-Modal Graph Topic Modeling. https://github.com/SmartVisions-AI/tritopic\n"
