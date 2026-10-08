"""
TriTopic adapter for the turftopic ecosystem.
==============================================

Provides :class:`TriTopicModel`, a wrapper around :class:`~tritopic.TriTopic`
that implements turftopic's ``ContextualModel`` interface.  This enables
TriTopic to participate in turftopic's shared tooling (``print_topics``,
``to_disk``, ``topicwizard`` visualisation, etc.) while exposing all native
TriTopic features (hierarchy, divide, overlap, metadata, ...).

Usage
-----
>>> from tritopic import TriTopicModel
>>> model = TriTopicModel(n_components=5)
>>> doc_topic = model.fit_transform(documents)
>>> model.print_topics()          # turftopic inherited
>>> model.build_hierarchy()       # TriTopic native
"""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np

try:
    from turftopic.base import ContextualModel

    _TURFTOPIC_AVAILABLE = True
except ImportError:
    from sklearn.base import BaseEstimator, TransformerMixin

    class ContextualModel(BaseEstimator, TransformerMixin):  # type: ignore[no-redef]
        """Minimal stand-in when turftopic is not installed."""

        pass

    _TURFTOPIC_AVAILABLE = False

from tritopic.core.model import TriTopic, TriTopicConfig, TopicInfo
from tritopic.core.hierarchy import TopicHierarchy


class TriTopicModel(ContextualModel):
    """Turftopic-compatible adapter around :class:`~tritopic.TriTopic`.

    Composition pattern: an internal ``TriTopic`` instance (accessible via
    :pyattr:`model_`) handles the actual modelling; this class translates
    between turftopic conventions (``components_``, ``classes_``,
    ``vectorizer``, ``encoder_``) and TriTopic's native API.

    Parameters
    ----------
    n_components : int or "auto"
        Number of topics.  ``"auto"`` lets Leiden decide.
    encoder : str or SentenceTransformer
        Sentence-transformers model name or instance.
    vectorizer : CountVectorizer or None
        Sklearn CountVectorizer for building the document-term matrix.
        If *None*, a default one is created.
    language : str
        Language for stopwords and keyword extraction.
    n_neighbors, graph_type, snn_weight, use_lexical_view, semantic_weight,
    lexical_weight, resolution, n_consensus_runs, min_cluster_size,
    use_iterative_refinement, max_iterations, convergence_threshold,
    use_dim_reduction, reduced_dims, dim_reduction_method, random_state,
    verbose
        Forwarded to :class:`~tritopic.TriTopicConfig`.
    """

    def __init__(
        self,
        n_components: int | Literal["auto"] = "auto",
        encoder: Union[str, Any] = "all-MiniLM-L6-v2",
        vectorizer: Any | None = None,
        # TriTopic-specific
        language: str = "english",
        n_neighbors: int = 15,
        graph_type: str = "hybrid",
        snn_weight: float = 0.5,
        use_lexical_view: bool = True,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        resolution: float = 1.0,
        n_consensus_runs: int = 10,
        min_cluster_size: int = 5,
        use_iterative_refinement: bool = True,
        max_iterations: int = 5,
        convergence_threshold: float = 0.95,
        use_dim_reduction: bool = True,
        reduced_dims: int = 10,
        dim_reduction_method: str = "umap",
        random_state: int = 42,
        verbose: bool = True,
    ):
        # Store every parameter as instance attribute (sklearn convention)
        self.n_components = n_components
        self.encoder = encoder
        self.language = language
        self.n_neighbors = n_neighbors
        self.graph_type = graph_type
        self.snn_weight = snn_weight
        self.use_lexical_view = use_lexical_view
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.resolution = resolution
        self.n_consensus_runs = n_consensus_runs
        self.min_cluster_size = min_cluster_size
        self.use_iterative_refinement = use_iterative_refinement
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_dim_reduction = use_dim_reduction
        self.reduced_dims = reduced_dims
        self.dim_reduction_method = dim_reduction_method
        self.random_state = random_state
        self.verbose = verbose

        # Vectorizer
        if vectorizer is None:
            from sklearn.feature_extraction.text import CountVectorizer

            self.vectorizer = CountVectorizer(
                min_df=2,
                max_df=0.95,
                stop_words="english" if language == "english" else None,
            )
        else:
            self.vectorizer = vectorizer

        # Encoder (turftopic expects self.encoder_ to be the actual encoder)
        if isinstance(encoder, str):
            self.encoder_ = _LazyEncoder(encoder)
        else:
            self.encoder_ = encoder

        # Will be set after fit
        self._model: TriTopic | None = None
        self.doc_term_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Turftopic required interface
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit the model and return document-topic probability matrix.

        Parameters
        ----------
        raw_documents : iterable of str
        y : ignored
        embeddings : ndarray, optional
            Pre-computed document embeddings.

        Returns
        -------
        doc_topic_matrix : ndarray of shape (n_documents, n_topics)
        """
        documents = list(raw_documents)

        # Build TriTopic config from stored parameters
        encoder_name = self.encoder if isinstance(self.encoder, str) else "custom"
        config = TriTopicConfig(
            embedding_model=encoder_name if isinstance(self.encoder, str) else "all-MiniLM-L6-v2",
            language=self.language,
            n_neighbors=self.n_neighbors,
            graph_type=self.graph_type,
            snn_weight=self.snn_weight,
            use_lexical_view=self.use_lexical_view,
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight,
            resolution=self.resolution,
            n_consensus_runs=self.n_consensus_runs,
            min_cluster_size=self.min_cluster_size,
            use_iterative_refinement=self.use_iterative_refinement,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            use_dim_reduction=self.use_dim_reduction,
            reduced_dims=self.reduced_dims,
            dim_reduction_method=self.dim_reduction_method,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        n_topics = self.n_components if self.n_components != "auto" else "auto"
        self._model = TriTopic(config=config, n_topics=n_topics)

        # If encoder is an actual SentenceTransformer, inject it
        if not isinstance(self.encoder, str):
            self._model._embedding_engine._model = self.encoder

        # Fit the native model
        self._model.fit(documents, embeddings=embeddings)

        # Build document-term matrix for c-TF-IDF
        self.doc_term_matrix = self.vectorizer.fit_transform(documents)

        # Compute c-TF-IDF components
        self._compute_components()

        # Set encoder_ to the actual model used
        self.encoder_ = self._model._embedding_engine._model

        return self._model.probabilities_

    def transform(
        self,
        raw_documents,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Infer topic probabilities for new documents.

        Parameters
        ----------
        raw_documents : iterable of str
        embeddings : ignored (for API compatibility)

        Returns
        -------
        doc_topic_matrix : ndarray of shape (n_documents, n_topics)
        """
        return self._model.transform_proba(list(raw_documents))

    # ------------------------------------------------------------------
    # c-TF-IDF components
    # ------------------------------------------------------------------

    def _compute_components(self) -> None:
        """Compute c-TF-IDF ``components_`` matrix over the full vocabulary."""
        labels = self._model.labels_
        topic_ids = sorted(
            [t.topic_id for t in self._model.topics_ if t.topic_id != -1]
        )
        n_vocab = self.doc_term_matrix.shape[1]

        # IDF
        doc_freq = np.asarray((self.doc_term_matrix > 0).sum(axis=0)).ravel()
        idf = np.log(len(labels) / (1 + doc_freq))

        components = np.zeros((len(topic_ids), n_vocab))
        for i, tid in enumerate(topic_ids):
            mask = labels == tid
            topic_tf = np.asarray(self.doc_term_matrix[mask].sum(axis=0)).ravel()
            topic_tf_norm = topic_tf / (topic_tf.sum() + 1e-10)
            components[i] = topic_tf_norm * idf

        self.components_ = components
        self.classes_ = np.array(topic_ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_(self) -> TriTopic:
        """Access the underlying native TriTopic instance."""
        if self._model is None:
            raise AttributeError("Model not fitted yet. Call fit_transform() first.")
        return self._model

    @property
    def topics_(self) -> list[TopicInfo]:
        return self.model_.topics_

    @property
    def labels_(self) -> np.ndarray:
        return self.model_.labels_

    @property
    def probabilities_(self) -> np.ndarray:
        return self.model_.probabilities_

    @property
    def hierarchy_(self) -> TopicHierarchy | None:
        return self.model_.hierarchy_

    @property
    def topic_names(self) -> list[str]:
        """Topic names for turftopic compatibility.

        Uses TriTopic labels if available, otherwise auto-generates
        from top keywords.
        """
        names = []
        for t in self.model_.topics_:
            if t.topic_id == -1:
                continue
            if t.label:
                names.append(t.label)
            else:
                top_words = "_".join(t.keywords[:4])
                names.append(f"{t.topic_id}_{top_words}")
        return names

    # ------------------------------------------------------------------
    # TriTopic feature delegation
    # ------------------------------------------------------------------

    def build_hierarchy(
        self,
        resolution_levels: list[float] | None = None,
        n_levels: int = 3,
    ) -> TopicHierarchy:
        """Build a multi-resolution topic hierarchy.

        Delegates to :meth:`TriTopic.build_hierarchy`.
        """
        return self.model_.build_hierarchy(
            resolution_levels=resolution_levels,
            n_levels=n_levels,
        )

    def divide(self, topic_id: int, n_subtopics: int = 2) -> list[TopicInfo]:
        """Split a topic into sub-topics.

        Delegates to :meth:`TriTopic.divide` and refreshes ``components_``.
        """
        result = self.model_.divide(topic_id, n_subtopics=n_subtopics)
        self._compute_components()
        return result

    def visualize_hierarchy_tree(self, **kwargs):
        """Visualize the topic hierarchy as a tree diagram."""
        return self.model_.visualize_hierarchy_tree(**kwargs)

    def get_document_topics(
        self,
        doc_idx: int,
        top_n: int = 3,
        method: Literal["centroid", "graph"] | None = None,
    ) -> list[tuple[int, float]]:
        """Return top-N topics with probabilities for a single document."""
        return self.model_.get_document_topics(doc_idx, top_n=top_n, method=method)

    def topic_overlap_matrix(self, threshold: float = 0.1):
        """Compute topic co-occurrence matrix from soft assignments."""
        return self.model_.topic_overlap_matrix(threshold=threshold)

    def visualize_overlap(self, threshold: float = 0.1, **kwargs):
        """Visualize the topic overlap matrix as a heatmap."""
        return self.model_.visualize_overlap(threshold=threshold, **kwargs)

    def reduce_outliers(
        self,
        strategy: Literal["embeddings", "neighbors"] = "embeddings",
        threshold: float | None = None,
    ) -> "TriTopicModel":
        """Reassign outlier documents. Refreshes ``components_``."""
        self.model_.reduce_outliers(strategy=strategy, threshold=threshold)
        self._compute_components()
        return self

    def reduce_topics(self, n_topics: int) -> "TriTopicModel":
        """Merge topics until *n_topics* remain. Refreshes ``components_``."""
        self.model_.reduce_topics(n_topics)
        self._compute_components()
        return self

    def merge_topics(self, topics_to_merge: list[int]) -> "TriTopicModel":
        """Merge specific topics. Refreshes ``components_``."""
        self.model_.merge_topics(topics_to_merge)
        self._compute_components()
        return self

    def generate_labels(self, labeler, topics: list[int] | None = None) -> None:
        """Generate LLM-powered topic labels."""
        self.model_.generate_labels(labeler, topics=topics)

    def evaluate(self) -> dict[str, float]:
        """Evaluate topic model quality."""
        return self.model_.evaluate()

    def save(self, path: str) -> None:
        """Save native TriTopic model to disk."""
        self.model_.save(path)

    @classmethod
    def load(cls, path: str) -> "TriTopicModel":
        """Load a TriTopicModel from a native TriTopic save file.

        Reconstructs the adapter around the loaded TriTopic instance.
        The ``components_`` are **not** restored (no doc-term matrix on
        disk); call ``fit_transform`` again if you need turftopic features
        like ``print_topics``.
        """
        native = TriTopic.load(path)
        cfg = native.config
        wrapper = cls(
            n_components=native.n_topics,
            encoder=cfg.embedding_model,
            language=cfg.language,
            n_neighbors=cfg.n_neighbors,
            graph_type=cfg.graph_type,
            snn_weight=cfg.snn_weight,
            use_lexical_view=cfg.use_lexical_view,
            semantic_weight=cfg.semantic_weight,
            lexical_weight=cfg.lexical_weight,
            resolution=cfg.resolution,
            n_consensus_runs=cfg.n_consensus_runs,
            min_cluster_size=cfg.min_cluster_size,
            use_iterative_refinement=cfg.use_iterative_refinement,
            max_iterations=cfg.max_iterations,
            convergence_threshold=cfg.convergence_threshold,
            use_dim_reduction=cfg.use_dim_reduction,
            reduced_dims=cfg.reduced_dims,
            dim_reduction_method=cfg.dim_reduction_method,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )
        wrapper._model = native
        return wrapper

    def __repr__(self) -> str:
        if self._model is not None and self._model._is_fitted:
            n = len([t for t in self._model.topics_ if t.topic_id != -1])
            return f"TriTopicModel(n_topics={n}, fitted=True)"
        return f"TriTopicModel(n_components={self.n_components!r}, fitted=False)"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class _LazyEncoder:
    """Placeholder that satisfies turftopic's ``encoder_`` attribute
    requirement before ``fit_transform`` has been called.

    Once TriTopic fits, the real SentenceTransformer replaces this.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    def encode(self, texts, **kwargs):
        from sentence_transformers import SentenceTransformer

        real = SentenceTransformer(self._model_name)
        return real.encode(texts, **kwargs)

    def __repr__(self):
        return f"_LazyEncoder({self._model_name!r})"
