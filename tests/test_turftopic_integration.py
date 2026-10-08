"""Tests for the turftopic integration adapter."""

import numpy as np
import pytest
from sklearn.base import clone

from tritopic.integrations.turftopic import TriTopicModel, _TURFTOPIC_AVAILABLE


@pytest.fixture(scope="module")
def documents():
    """30 short synthetic documents spanning 3 rough themes."""
    return [
        # Theme A: space / astronomy
        "NASA launched a new satellite into orbit around Mars",
        "The Hubble telescope captured images of a distant galaxy",
        "Astronomers discovered a new exoplanet in the habitable zone",
        "SpaceX successfully landed its booster after launch",
        "The International Space Station orbits Earth every 90 minutes",
        "Mars rover collected soil samples from the red planet",
        "A solar eclipse was visible across North America yesterday",
        "Scientists detected gravitational waves from merging black holes",
        "The Artemis mission aims to return humans to the Moon",
        "Jupiter has over 80 known moons orbiting the gas giant",
        # Theme B: sports / baseball
        "The pitcher threw a perfect game with 27 consecutive outs",
        "Home run records were broken during the baseball season",
        "The World Series attracted millions of viewers this year",
        "Spring training begins in February for all baseball teams",
        "The shortstop made an incredible diving catch in the ninth",
        "Baseball players negotiated a new collective bargaining agreement",
        "The minor league team won the championship last night",
        "Batting averages declined across the league this season",
        "The umpire called a controversial strike three to end the game",
        "Free agent signings dominated the baseball off-season news",
        # Theme C: medicine / health
        "A new vaccine was approved for the prevention of malaria",
        "Clinical trials showed promising results for the cancer drug",
        "Researchers published a study on the effects of sleep deprivation",
        "The hospital implemented a new protocol for emergency surgery",
        "Antibiotics resistance is a growing concern among health experts",
        "A breakthrough in gene therapy offers hope for rare diseases",
        "The flu season peaked earlier than expected this winter",
        "Doctors recommend regular exercise to prevent heart disease",
        "Mental health awareness campaigns are increasing worldwide",
        "A new diagnostic tool detects infections within minutes",
    ]


@pytest.fixture(scope="module")
def embeddings(documents):
    """Random embeddings matching documents, deterministic seed."""
    rng = np.random.RandomState(42)
    emb = rng.randn(len(documents), 64).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture(scope="module")
def fitted_model(documents, embeddings):
    """A TriTopicModel fitted on fake data (fast, no real embeddings)."""
    model = TriTopicModel(
        n_components="auto",
        verbose=False,
        n_neighbors=5,
        use_dim_reduction=False,
        use_iterative_refinement=False,
        n_consensus_runs=3,
        min_cluster_size=2,
    )
    model.fit_transform(documents, embeddings=embeddings)
    return model


# ------------------------------------------------------------------
# Interface compliance
# ------------------------------------------------------------------

class TestFitTransform:
    def test_returns_2d_array(self, fitted_model, documents):
        """fit_transform returns (n_docs, n_topics) array."""
        # Re-run to capture return value
        doc_topic = fitted_model.probabilities_
        assert doc_topic.ndim == 2
        assert doc_topic.shape[0] == len(documents)

    def test_rows_sum_to_one(self, fitted_model):
        """Probability rows should sum to ~1.0."""
        row_sums = fitted_model.probabilities_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_non_negative(self, fitted_model):
        """Probabilities should be non-negative."""
        assert np.all(fitted_model.probabilities_ >= 0)


class TestRequiredAttributes:
    def test_components_exists(self, fitted_model):
        assert hasattr(fitted_model, "components_")
        assert isinstance(fitted_model.components_, np.ndarray)

    def test_components_shape(self, fitted_model):
        """components_ is (n_topics, n_vocab)."""
        n_topics = len([t for t in fitted_model.topics_ if t.topic_id != -1])
        n_vocab = fitted_model.doc_term_matrix.shape[1]
        assert fitted_model.components_.shape == (n_topics, n_vocab)

    def test_components_non_negative(self, fitted_model):
        """c-TF-IDF values should be non-negative."""
        assert np.all(fitted_model.components_ >= 0)

    def test_components_rows_have_values(self, fitted_model):
        """Each topic row should have non-zero entries."""
        for i in range(fitted_model.components_.shape[0]):
            assert np.any(fitted_model.components_[i] > 0)

    def test_classes_exists(self, fitted_model):
        assert hasattr(fitted_model, "classes_")
        assert isinstance(fitted_model.classes_, np.ndarray)

    def test_encoder_exists(self, fitted_model):
        assert hasattr(fitted_model, "encoder_")

    def test_vectorizer_exists(self, fitted_model):
        assert hasattr(fitted_model, "vectorizer")

    def test_doc_term_matrix_exists(self, fitted_model):
        assert hasattr(fitted_model, "doc_term_matrix")
        assert fitted_model.doc_term_matrix is not None


class TestTransform:
    @pytest.mark.skipif(
        True,
        reason="transform() encodes new docs with the real model; "
        "dimension mismatch when fitted with fake embeddings",
    )
    def test_transform_shape(self, fitted_model):
        """transform() returns (n_new_docs, n_topics)."""
        new_docs = [
            "A new planet was discovered orbiting a nearby star",
            "The baseball team signed a major free agent",
        ]
        result = fitted_model.transform(new_docs)
        n_topics = len([t for t in fitted_model.topics_ if t.topic_id != -1])
        assert result.shape == (2, n_topics)

    @pytest.mark.skipif(
        True,
        reason="transform() encodes new docs with the real model; "
        "dimension mismatch when fitted with fake embeddings",
    )
    def test_transform_rows_sum_to_one(self, fitted_model):
        new_docs = ["Research on antibiotic resistance continues"]
        result = fitted_model.transform(new_docs)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=0.01)


# ------------------------------------------------------------------
# sklearn compatibility
# ------------------------------------------------------------------

class TestSklearn:
    def test_get_params(self, fitted_model):
        params = fitted_model.get_params()
        assert "n_components" in params
        assert "language" in params
        assert "n_neighbors" in params

    def test_set_params(self):
        model = TriTopicModel(n_components=5)
        model.set_params(n_neighbors=20)
        assert model.n_neighbors == 20

    def test_clone(self):
        model = TriTopicModel(n_components=3, n_neighbors=10)
        cloned = clone(model)
        assert cloned.n_components == 3
        assert cloned.n_neighbors == 10
        assert cloned._model is None  # not fitted


# ------------------------------------------------------------------
# TriTopic feature delegation
# ------------------------------------------------------------------

class TestDelegation:
    def test_properties(self, fitted_model):
        assert fitted_model.topics_ is not None
        assert fitted_model.labels_ is not None
        assert fitted_model.probabilities_ is not None

    def test_model_property(self, fitted_model):
        from tritopic import TriTopic
        assert isinstance(fitted_model.model_, TriTopic)

    def test_topic_names(self, fitted_model):
        names = fitted_model.topic_names
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_build_hierarchy(self, fitted_model):
        from tritopic.core.hierarchy import TopicHierarchy
        h = fitted_model.build_hierarchy(n_levels=2)
        assert isinstance(h, TopicHierarchy)
        assert fitted_model.hierarchy_ is not None

    def test_evaluate(self, fitted_model):
        metrics = fitted_model.evaluate()
        assert "coherence_mean" in metrics
        assert "diversity" in metrics
        assert "n_topics" in metrics

    def test_get_document_topics(self, fitted_model):
        result = fitted_model.get_document_topics(0, top_n=2)
        assert isinstance(result, list)
        assert len(result) <= 2
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)


class TestReduceTopics:
    def test_reduce_updates_components(self, documents, embeddings):
        """reduce_topics should update components_ dimensions."""
        model = TriTopicModel(
            verbose=False,
            n_neighbors=5,
            use_dim_reduction=False,
            use_iterative_refinement=False,
            n_consensus_runs=3,
            min_cluster_size=2,
        )
        model.fit_transform(documents, embeddings=embeddings)
        n_before = model.components_.shape[0]
        if n_before <= 2:
            pytest.skip("Need more than 2 topics to test reduce")
        model.reduce_topics(2)
        assert model.components_.shape[0] == 2
        assert model.classes_.shape[0] == 2


class TestDivide:
    def test_divide_updates_components(self, documents, embeddings):
        """divide should update components_ after splitting."""
        model = TriTopicModel(
            verbose=False,
            n_neighbors=5,
            use_dim_reduction=False,
            use_iterative_refinement=False,
            n_consensus_runs=3,
            min_cluster_size=2,
        )
        model.fit_transform(documents, embeddings=embeddings)
        non_outlier = [t for t in model.topics_ if t.topic_id != -1]
        if not non_outlier:
            pytest.skip("No non-outlier topics")
        tid = non_outlier[0].topic_id
        n_before = model.components_.shape[0]
        model.divide(tid, n_subtopics=2)
        # After divide, should have at least as many topics
        assert model.components_.shape[0] >= n_before


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, fitted_model, tmp_path):
        """save/load preserves the native TriTopic state."""
        path = str(tmp_path / "model.pkl")
        fitted_model.save(path)

        loaded = TriTopicModel.load(path)
        assert loaded._model is not None
        assert loaded._model._is_fitted
        assert len(loaded.topics_) == len(fitted_model.topics_)
        np.testing.assert_array_equal(loaded.labels_, fitted_model.labels_)


# ------------------------------------------------------------------
# Turftopic-specific (conditional)
# ------------------------------------------------------------------

@pytest.mark.skipif(not _TURFTOPIC_AVAILABLE, reason="turftopic not installed")
class TestTurftopicSpecific:
    def test_inherits_contextual_model(self, fitted_model):
        from turftopic.base import ContextualModel
        assert isinstance(fitted_model, ContextualModel)

    def test_get_vocab(self, fitted_model):
        vocab = fitted_model.get_vocab()
        assert len(vocab) == fitted_model.components_.shape[1]

    def test_print_topics(self, fitted_model, capsys):
        """print_topics should run without error."""
        fitted_model.print_topics(top_k=5)
        # Just verify it didn't crash


class TestRepr:
    def test_unfitted_repr(self):
        m = TriTopicModel(n_components=5)
        assert "fitted=False" in repr(m)

    def test_fitted_repr(self, fitted_model):
        assert "fitted=True" in repr(fitted_model)
