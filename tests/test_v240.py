"""Regression tests for the 2.4.0 changes (performance + correctness fixes)."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from tritopic import TriTopic, TriTopicConfig
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.graph_builder import GraphBuilder, MetadataFeatures
from tritopic.core.keywords import KeywordExtractor
from tritopic.utils.metrics import coherence_from_doc_term, compute_coherence_batch
from tritopic.utils.stopwords import TOKEN_PATTERN

THEMES = {
    "space": "orbit rocket nasa planet galaxy telescope astronaut launch moon satellite",
    "baseball": "pitcher inning homerun batter umpire league season stadium shortstop catcher",
    "medicine": "vaccine patient clinical therapy hospital doctor disease surgery diagnosis treatment",
}


@pytest.fixture(scope="module")
def clustered():
    """120 docs in 3 well-separated themes with matching clustered embeddings."""
    rng = np.random.default_rng(0)
    docs, labels, centers = [], [], rng.normal(size=(3, 32)) * 4
    for t, (name, vocab) in enumerate(THEMES.items()):
        words = vocab.split()
        for _ in range(40):
            docs.append(" ".join(rng.choice(words, 8)) + " the and of")
            labels.append(t)
    labels = np.array(labels)
    emb = centers[labels] + rng.normal(size=(len(docs), 32))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return docs, labels, emb


def _fit(docs, emb, **cfg):
    config = TriTopicConfig(verbose=False, use_dim_reduction=False, n_neighbors=10, **cfg)
    return TriTopic(config=config).fit(docs, embeddings=emb)


# --- clustering -----------------------------------------------------------

def test_auto_resolution_recovers_themes(clustered):
    docs, labels, emb = clustered
    model = _fit(docs, emb)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(labels, model.labels_) > 0.9
    assert model.resolution_search_, "auto mode should record its resolution scan"
    assert model.resolution_ in [r for r, _, _ in model.resolution_search_]


def test_auto_resolution_can_be_disabled(clustered):
    docs, _, emb = clustered
    model = _fit(docs, emb, auto_resolution=False, resolution=0.7)
    assert model.resolution_ == 0.7
    assert model.resolution_search_ == []


def test_n_topics_exact(clustered):
    docs, _, emb = clustered
    config = TriTopicConfig(verbose=False, use_dim_reduction=False, n_neighbors=10)
    model = TriTopic(config=config, n_topics=5).fit(docs, embeddings=emb)
    assert len([t for t in model.topics_ if t.topic_id != -1]) == 5


def test_consensus_identical_partitions_shortcut():
    import igraph as ig
    g = ig.Graph.Famous("Zachary")
    g.es["weight"] = [1.0] * g.ecount()
    clusterer = ConsensusLeiden(n_runs=4, resolution=1.0)
    p = np.arange(g.vcount()) % 2
    assert np.array_equal(clusterer._compute_consensus(g, [p, p, p], 1.0), p)


def test_find_optimal_resolution_counts_only_real_topics():
    import igraph as ig
    # 4 cliques of 10 plus 10 isolated nodes: isolated nodes must not count as topics
    edges = [(c * 10 + i, c * 10 + j) for c in range(4) for i in range(10) for j in range(i + 1, 10)]
    g = ig.Graph(n=50, edges=edges)
    g.es["weight"] = [1.0] * g.ecount()
    clusterer = ConsensusLeiden()
    res = clusterer.find_optimal_resolution(g, (0.01, 5.0), n_steps=12, target_n_topics=4, min_cluster_size=5)
    labels = clusterer.fit_predict(g, min_cluster_size=5, resolution=res)
    assert len(set(labels) - {-1}) == 4


# --- graph ----------------------------------------------------------------

def test_snn_matches_set_intersection(clustered):
    _, _, emb = clustered
    gb = GraphBuilder(n_neighbors=6)
    _, idx, _ = gb._compute_knn(emb)
    snn = gb.build_snn_graph(emb, _precomputed_indices=idx).toarray()
    sets = [set(row[1:]) for row in idx]
    i, j = 3, idx[3][1]
    assert snn[i, j] == pytest.approx(len(sets[i] & sets[j]) / 6)


def test_euclidean_kernel_weights_in_unit_interval():
    rng = np.random.default_rng(1)
    gb = GraphBuilder(n_neighbors=5, metric="euclidean")
    _, _, sims = gb._compute_knn(rng.normal(size=(50, 3)) * 100)  # scale-free kernel
    assert sims.min() >= 0 and sims.max() <= 1
    assert sims[:, 1:].mean() > 0.1  # not collapsed to ~0 like 1/(1+d) at large scale


def test_metadata_only_reweights_existing_edges(clustered):
    docs, _, emb = clustered
    gb = GraphBuilder(n_neighbors=5)
    meta = gb.build_metadata_graph(pd.DataFrame({
        "source": pd.Series(["a", "b", "c"] * 40, dtype="string"),  # pandas string dtype
        "score": np.linspace(0, 1, 120),
        "flag": [True, False] * 60,
    }))
    assert isinstance(meta, MetadataFeatures)
    weights = {"semantic": 0.5, "lexical": 0.0, "metadata": 0.2}
    g_plain = gb.build_multiview_graph(emb, weights={"semantic": 1.0})
    g_meta = gb.build_multiview_graph(emb, metadata_graph=meta, weights=weights)
    assert g_meta.ecount() == g_plain.ecount()  # no clique edges added


def test_lexical_from_counts_matches_vectorizer(clustered):
    docs, _, _ = clustered
    kx = KeywordExtractor()
    gb = GraphBuilder()
    from_counts = gb.build_lexical_matrix_from_counts(kx.fit_corpus(docs))
    assert from_counts.shape[0] == len(docs)
    assert np.allclose(np.asarray(from_counts.multiply(from_counts).sum(axis=1)).ravel(), 1.0)


# --- keywords & metrics ---------------------------------------------------

def test_token_pattern_drops_numbers_and_underscores():
    import re
    tokens = re.findall(TOKEN_PATTERN, "mr2 covid19 1993 000 __init__ data")
    assert tokens == ["mr2", "covid19", "data"]


def test_ctfidf_keywords_match_theme(clustered):
    docs, labels, _ = clustered
    kw = KeywordExtractor().extract_all_topics(docs, labels, n_keywords=5)
    for t, vocab in enumerate(THEMES.values()):
        assert len(set(kw[t][0]) & set(vocab.split())) >= 4


def test_bm25_uses_topic_documents(clustered):
    docs, labels, _ = clustered
    kx = KeywordExtractor(method="bm25")
    kw = kx.extract_all_topics(docs, labels, n_keywords=5)
    # topic 2 (medicine) is the *last* third of the corpus; the old bug used the first n docs
    assert len(set(kw[2][0]) & set(THEMES["medicine"].split())) >= 4
    single = kx.extract([d for d, l in zip(docs, labels) if l == 2], docs, 5)
    assert single[0] == kw[2][0]


def test_coherence_corpus_reference_and_bigrams():
    docs = ["space shuttle launch", "space shuttle orbit", "baseball pitcher", "baseball inning"]
    good, bad = compute_coherence_batch([["space", "shuttle"], ["space", "baseball"]], docs)
    assert good > 0.9 and bad == -1.0
    assert compute_coherence_batch([["space shuttle", "launch"]], docs)[0] > 0  # bigram counted


def test_coherence_from_doc_term_matches_batch(clustered):
    docs, labels, _ = clustered
    kx = KeywordExtractor()
    X = kx.fit_corpus(docs)
    kw = kx.extract_all_topics(docs, labels, n_keywords=5)
    index = {w: i for i, w in enumerate(kx._vocabulary)}
    ids = [[index[w] for w in kw[t][0]] for t in sorted(kw)]
    a = coherence_from_doc_term(X, ids)
    b = compute_coherence_batch([kw[t][0] for t in sorted(kw)], docs)
    assert np.allclose(a, b)


# --- post-fit operations --------------------------------------------------

def test_divide_respects_requested_count(clustered):
    docs, _, emb = clustered
    model = _fit(docs, emb, auto_resolution=False, resolution=0.05)
    biggest = max((t for t in model.topics_ if t.topic_id != -1), key=lambda t: t.size)
    subs = model.divide(biggest.topic_id, n_subtopics=2)
    assert 1 <= len(subs) <= 3
    assert all(s is not None for s in subs)


def test_overlap_matrix_vectorized_matches_loop(clustered):
    docs, _, emb = clustered
    model = _fit(docs, emb)
    ov = model.topic_overlap_matrix(threshold=0.2).values
    active = model.probabilities_ >= 0.2
    expected = np.array([[np.sum(active[:, i] & active[:, j]) for j in range(active.shape[1])]
                         for i in range(active.shape[1])])
    assert np.array_equal(ov, expected)


def test_resolution_persists_through_save_load(clustered, tmp_path):
    docs, _, emb = clustered
    model = _fit(docs, emb)
    path = tmp_path / "m.pkl"
    model.save(str(path))
    loaded = TriTopic.load(str(path))
    assert loaded.resolution_ == model.resolution_
    assert loaded.resolution_search_ == model.resolution_search_


def test_auto_resolution_verbose_output(clustered, capsys):
    """Regression: verbose=True crashed in the auto-resolution progress line."""
    docs, _, emb = clustered
    TriTopic(config=TriTopicConfig(verbose=True, use_dim_reduction=False, n_neighbors=10)).fit(docs, embeddings=emb)
    assert "Auto resolution" in capsys.readouterr().out
