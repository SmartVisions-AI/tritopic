"""
TriTopic: Main Model Class
===========================

The core class that orchestrates all components of the topic modeling pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.graph_builder import GraphBuilder
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.keywords import KeywordExtractor
from tritopic.core.hierarchy import TopicNode, TopicHierarchy
from tritopic.utils.metrics import compute_coherence_batch, compute_diversity


@dataclass
class TopicInfo:
    """Container for topic information."""
    
    topic_id: int
    size: int
    keywords: list[str]
    keyword_scores: list[float]
    representative_docs: list[int]
    label: str | None = None
    description: str | None = None
    centroid: np.ndarray | None = None
    coherence: float | None = None


@dataclass
class TriTopicConfig:
    """Configuration for TriTopic model."""
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    language: str = "english"
    
    # Graph settings
    n_neighbors: int = 15
    metric: str = "cosine"
    graph_type: Literal["mutual_knn", "snn", "hybrid"] = "hybrid"
    snn_weight: float = 0.5
    
    # Multi-view settings
    use_lexical_view: bool = True
    use_metadata_view: bool = False
    lexical_weight: float = 0.3
    metadata_weight: float = 0.2
    semantic_weight: float = 0.5
    
    # Clustering settings
    # Leiden resolution (higher = more topics).  Used when n_topics is given
    # as the starting point of the search, and for n_topics="auto" when
    # auto_resolution is False.  1.0 over-segments kNN graphs heavily.
    resolution: float = 0.3
    # With n_topics="auto": scan resolutions in resolution_range and pick the
    # coarsest one whose mean keyword coherence (NPMI) is within
    # auto_resolution_tolerance of the best, among partitions where no topic
    # holds more than auto_resolution_max_share of the documents (very coarse
    # partitions get high NPMI from generic co-occurring words).  Adapts the
    # topic count to the corpus (dev benchmarks, 3 seeds: NMI 0.565 -> 0.594,
    # ARI 0.415 -> 0.474 vs. a fixed resolution of 0.3).
    auto_resolution: bool = True
    resolution_range: tuple[float, float] | None = None  # default (0.01, 1.0)
    auto_resolution_steps: int = 15
    auto_resolution_tolerance: float = 0.05
    auto_resolution_max_share: float = 0.5
    n_consensus_runs: int = 10
    min_cluster_size: int = 5
    
    # Iterative refinement
    use_iterative_refinement: bool = True
    max_iterations: int = 5
    convergence_threshold: float = 0.95
    
    # Keyword extraction
    n_keywords: int = 10
    n_representative_docs: int = 5
    keyword_method: Literal["ctfidf", "bm25", "keybert"] = "ctfidf"
    
    # Dimensionality reduction
    use_dim_reduction: bool = True
    reduced_dims: int = 10
    dim_reduction_method: Literal["umap", "pacmap"] = "umap"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0  # 0.0 for clustering (not visualization)
    # Metric for the kNN graph on *reduced* embeddings.  UMAP/PaCMAP output is
    # a Euclidean layout; cosine (angle around the arbitrary origin) would
    # merge clusters that happen to lie on the same ray.
    reduced_metric: str = "euclidean"

    # Outlier handling
    outlier_threshold: float = 0.35

    # Soft assignment
    soft_assignment_method: Literal["centroid", "graph"] = "centroid"

    # Probability temperature (higher -> sharper distributions)
    softmax_temperature: float = 5.0

    # Misc
    random_state: int = 42
    verbose: bool = True


class TriTopic:
    """
    Tri-Modal Graph Topic Modeling with Iterative Refinement.
    
    A state-of-the-art topic modeling approach that combines semantic embeddings,
    lexical similarity, and optional metadata to create robust, interpretable topics.
    
    Key innovations:
    - Multi-view graph fusion (semantic + lexical + metadata)
    - Leiden clustering with consensus for stability
    - Iterative refinement loop for optimal topic separation
    - Advanced keyword extraction with representative documents
    - Optional LLM-powered topic labeling
    
    Parameters
    ----------
    config : TriTopicConfig, optional
        Configuration object. If None, uses defaults.
    embedding_model : str, optional
        Name of sentence-transformers model. Default: "all-MiniLM-L6-v2"
    n_neighbors : int, optional
        Number of neighbors for graph construction. Default: 15
    n_topics : int or "auto", optional
        Number of topics. "auto" picks the resolution with the most coherent
        topics (see ``TriTopicConfig.auto_resolution``). Default: "auto"
    use_iterative_refinement : bool, optional
        Whether to use the iterative refinement loop. Default: True
    verbose : bool, optional
        Print progress information. Default: True
    
    Attributes
    ----------
    topics_ : list[TopicInfo]
        Information about each discovered topic.
    labels_ : np.ndarray
        Topic assignment for each document.
    embeddings_ : np.ndarray
        Document embeddings.
    graph_ : igraph.Graph
        The constructed similarity graph.
    topic_embeddings_ : np.ndarray
        Centroid embeddings for each topic.
    
    Examples
    --------
    Basic usage:
    
    >>> from tritopic import TriTopic
    >>> model = TriTopic(n_neighbors=15, verbose=True)
    >>> topics = model.fit_transform(documents)
    >>> print(model.get_topic_info())
    
    With metadata:
    
    >>> model = TriTopic()
    >>> model.config.use_metadata_view = True
    >>> topics = model.fit_transform(documents, metadata=df[['source', 'date']])
    
    With LLM labeling:
    
    >>> from tritopic import TriTopic, LLMLabeler
    >>> model = TriTopic()
    >>> model.fit_transform(documents)
    >>> labeler = LLMLabeler(provider="anthropic", api_key="...")
    >>> model.generate_labels(labeler)
    """
    
    def __init__(
        self,
        config: TriTopicConfig | None = None,
        embedding_model: str | None = None,
        n_neighbors: int | None = None,
        n_topics: int | Literal["auto"] = "auto",
        use_iterative_refinement: bool | None = None,
        language: str | None = None,
        verbose: bool | None = None,
        random_state: int | None = None,
    ):
        # Initialize config
        self.config = config or TriTopicConfig()

        # Override config with explicit parameters
        if embedding_model is not None:
            self.config.embedding_model = embedding_model
        if n_neighbors is not None:
            self.config.n_neighbors = n_neighbors
        if use_iterative_refinement is not None:
            self.config.use_iterative_refinement = use_iterative_refinement
        if language is not None:
            self.config.language = language
        if verbose is not None:
            self.config.verbose = verbose
        if random_state is not None:
            self.config.random_state = random_state

        # Auto-select multilingual embedding model
        if self.config.language == "multilingual" and self.config.embedding_model == "all-MiniLM-L6-v2":
            self.config.embedding_model = "BAAI/bge-m3"

        self.n_topics = n_topics

        # Initialize components
        self._embedding_engine = EmbeddingEngine(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )
        self._graph_builder = GraphBuilder(
            n_neighbors=self.config.n_neighbors,
            metric=self.config.metric,
            graph_type=self.config.graph_type,
            snn_weight=self.config.snn_weight,
            language=self.config.language,
        )
        self._clusterer = ConsensusLeiden(
            resolution=self.config.resolution,
            n_runs=self.config.n_consensus_runs,
            random_state=self.config.random_state,
        )
        self._keyword_extractor = KeywordExtractor(
            method=self.config.keyword_method,
            n_keywords=self.config.n_keywords,
            language=self.config.language,
        )
        
        # State
        self.topics_: list[TopicInfo] = []
        self.labels_: np.ndarray | None = None
        self.embeddings_: np.ndarray | None = None
        self.original_embeddings_: np.ndarray | None = None  # unrefined, for transform()
        self.reduced_embeddings_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
        self.lexical_matrix_: Any | None = None
        self.graph_: Any | None = None
        self.topic_embeddings_: np.ndarray | None = None
        self.documents_: list[str] | None = None
        self.hierarchy_: TopicHierarchy | None = None
        self._is_fitted: bool = False
        self._iteration_history: list[dict] = []
        self.resolution_: float = self.config.resolution
        self.resolution_search_: list[tuple[float, int, float]] = []
        self._dim_reducer: Any | None = None
        
    def fit(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> "TriTopic":
        """
        Fit the topic model to documents.
        
        Parameters
        ----------
        documents : list[str]
            List of document texts.
        embeddings : np.ndarray, optional
            Pre-computed embeddings. If None, computed automatically.
        metadata : pd.DataFrame, optional
            Document metadata for the metadata view.
            
        Returns
        -------
        self : TriTopic
            Fitted model.
        """
        # Input validation
        if not documents:
            raise ValueError("documents must be a non-empty list of strings.")
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match "
                f"documents length ({len(documents)})."
            )
        if metadata is not None and len(metadata) != len(documents):
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match "
                f"documents length ({len(documents)})."
            )

        self.documents_ = documents
        n_docs = len(documents)

        # Reset stateful components for clean re-fitting
        self._keyword_extractor.reset()
        self._iteration_history = []
        self.reduced_embeddings_ = None
        self._dim_reducer = None

        if self.config.verbose:
            print(f"[TriTopic] Fitting model on {n_docs} documents")
            print(f"   Config: {self.config.graph_type} graph, "
                  f"{'iterative' if self.config.use_iterative_refinement else 'single-pass'} mode")

        # Step 1: Generate embeddings
        if embeddings is not None:
            self.embeddings_ = embeddings
            if self.config.verbose:
                print("   + Using provided embeddings")
        else:
            if self.config.verbose:
                print(f"   > Generating embeddings ({self.config.embedding_model})...")
            self.embeddings_ = self._embedding_engine.encode(documents)

        # Keep unrefined copy so transform() compares new docs in the same space
        self.original_embeddings_ = self.embeddings_.copy()

        # Step 1.5: Dimensionality reduction for graph building
        if self.config.use_dim_reduction:
            self._reduce_dimensions()
        self._graph_builder.metric = (
            self.config.reduced_metric
            if self.reduced_embeddings_ is not None
            else self.config.metric
        )

        # Step 2: Build lexical representation
        if self.config.use_lexical_view:
            if self.config.verbose:
                print("   > Building lexical similarity matrix...")
            # One tokenization pass shared with keyword extraction
            doc_term = self._keyword_extractor.fit_corpus(documents)
            self.lexical_matrix_ = self._graph_builder.build_lexical_matrix_from_counts(doc_term)

        # Step 3: Build metadata graph (if provided)
        self._metadata_graph = None
        if self.config.use_metadata_view and metadata is not None:
            if self.config.verbose:
                print("   > Building metadata similarity graph...")
            self._metadata_graph = self._graph_builder.build_metadata_graph(metadata)

        # Step 3.5: Choose the Leiden resolution
        self.resolution_ = self.config.resolution
        if self.n_topics == "auto" and self.config.auto_resolution:
            self.resolution_ = self._select_resolution(documents, self._metadata_graph)
        self._clusterer.resolution = self.resolution_

        # Step 4: Main fitting loop
        if self.config.use_iterative_refinement:
            self._fit_iterative(documents, self._metadata_graph)
        else:
            self._fit_single_pass(documents, self._metadata_graph)

        # Step 5: Extract keywords and representative docs
        if self.config.verbose:
            print("   > Extracting keywords and representative documents...")
        self._extract_topic_info(documents)

        # Step 6: Compute topic centroids
        self._compute_topic_centroids()

        # Step 7: Compute soft assignments (probabilities)
        self._compute_probabilities()

        self._is_fitted = True

        # Step 8: Apply n_topics target if specified
        if self.n_topics != "auto" and isinstance(self.n_topics, int):
            current_n_topics = len([t for t in self.topics_ if t.topic_id != -1])
            if self.n_topics != current_n_topics:
                # Use resolution search in both directions (fewer or more topics)
                self._auto_resolve_topic_count(
                    documents, self._metadata_graph, current_n_topics
                )

        if self.config.verbose:
            n_topics = len([t for t in self.topics_ if t.topic_id != -1])
            n_outliers = np.sum(self.labels_ == -1) if self.labels_ is not None else 0
            print(f"\n[OK] Fitting complete!")
            print(f"   Found {n_topics} topics")
            print(f"   {n_outliers} outlier documents ({100*n_outliers/n_docs:.1f}%)")

        return self
    
    def _fit_single_pass(
        self,
        documents: list[str],
        metadata_graph: Any | None = None,
    ) -> None:
        """Single-pass fitting without iterative refinement."""
        # Build graph
        if self.config.verbose:
            print("   > Building multi-view graph...")

        # Use reduced embeddings for graph building if available
        graph_embeddings = self.reduced_embeddings_ if self.reduced_embeddings_ is not None else self.embeddings_

        self.graph_ = self._graph_builder.build_multiview_graph(
            semantic_embeddings=graph_embeddings,
            lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
            metadata_graph=metadata_graph,
            weights={
                "semantic": self.config.semantic_weight,
                "lexical": self.config.lexical_weight,
                "metadata": self.config.metadata_weight,
            }
        )
        
        # Cluster
        if self.config.verbose:
            print(f"   > Running Leiden consensus clustering ({self.config.n_consensus_runs} runs)...")
            
        self.labels_ = self._clusterer.fit_predict(
            self.graph_,
            min_cluster_size=self.config.min_cluster_size,
        )
    
    def _fit_iterative(
        self,
        documents: list[str],
        metadata_graph: Any | None = None,
    ) -> None:
        """Iterative refinement fitting loop."""
        if self.config.verbose:
            print(f"   > Starting iterative refinement (max {self.config.max_iterations} iterations)...")

        # Refinement happens in the space the graph is built from.  With dim
        # reduction that is the reduced space: refining the full embeddings and
        # re-projecting them with UMAP.transform() every iteration was the main
        # runtime cost and produced a different (noisier) layout than
        # fit_transform(), so iterations disagreed for reasons unrelated to
        # the refinement itself.
        use_reduced = self.reduced_embeddings_ is not None
        current = (self.reduced_embeddings_ if use_reduced else self.embeddings_).copy()
        refine_metric = "euclidean" if use_reduced else "cosine"
        previous_labels = None

        # Pre-compute lexical adjacency once (TF-IDF never changes across iterations)
        cached_lexical_adj = None
        if self.config.use_lexical_view and self.lexical_matrix_ is not None:
            cached_lexical_adj = self._graph_builder.build_lexical_graph(
                self.lexical_matrix_
            )

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"      Iteration {iteration + 1}...")

            self.graph_ = self._graph_builder.build_multiview_graph(
                semantic_embeddings=current,
                lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
                metadata_graph=metadata_graph,
                weights={
                    "semantic": self.config.semantic_weight,
                    "lexical": self.config.lexical_weight,
                    "metadata": self.config.metadata_weight,
                },
                lexical_adj=cached_lexical_adj,
            )

            # Cluster
            self.labels_ = self._clusterer.fit_predict(
                self.graph_,
                min_cluster_size=self.config.min_cluster_size,
            )

            n_topics_found = len(np.unique(self.labels_[self.labels_ != -1]))

            # Check convergence
            ari = None
            if previous_labels is not None:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(previous_labels, self.labels_)
                if self.config.verbose:
                    print(f"         ARI vs previous: {ari:.4f}")
            self._iteration_history.append({
                "iteration": iteration + 1,
                "ari": ari,
                "n_topics": n_topics_found,
            })

            if ari is not None and ari >= self.config.convergence_threshold:
                if self.config.verbose:
                    print(f"      Converged at iteration {iteration + 1}")
                break
            if iteration == self.config.max_iterations - 1:
                break  # no refinement needed after the last clustering

            previous_labels = self.labels_.copy()

            # Refine embeddings with decaying blend factor (aggressive->fine)
            blend = 0.3 - 0.2 * (iteration / max(self.config.max_iterations - 1, 1))
            current = self._refine_embeddings(
                current, self.labels_, blend_factor=blend, metric=refine_metric
            )

        # Store final refined embeddings
        if use_reduced:
            self.reduced_embeddings_ = current
        else:
            self.embeddings_ = current

    def _select_resolution(
        self,
        documents: list[str],
        metadata_graph: Any | None = None,
    ) -> float:
        """Pick the resolution whose topics have the most coherent keywords.

        Scans ``auto_resolution_steps`` resolutions (log-spaced over
        ``resolution_range``) with one Leiden run each on the initial graph,
        scores every partition by the mean NPMI of its c-TF-IDF keywords
        (computed from the cached document-term matrix, no re-tokenizing),
        and returns the coarsest resolution within
        ``auto_resolution_tolerance`` of the best score among partitions
        whose largest topic holds at most ``auto_resolution_max_share`` of
        the documents.  The scan is stored
        in ``resolution_search_`` as (resolution, n_topics, coherence).
        """
        import leidenalg as la
        from tritopic.utils.metrics import coherence_from_doc_term

        graph_embeddings = self.reduced_embeddings_ if self.reduced_embeddings_ is not None else self.embeddings_
        graph = self._graph_builder.build_multiview_graph(
            semantic_embeddings=graph_embeddings,
            lexical_matrix=self.lexical_matrix_ if self.config.use_lexical_view else None,
            metadata_graph=metadata_graph,
            weights={
                "semantic": self.config.semantic_weight,
                "lexical": self.config.lexical_weight,
                "metadata": self.config.metadata_weight,
            },
        )

        kx = self._keyword_extractor
        doc_term = kx.fit_corpus(documents)
        term_index = {term: i for i, term in enumerate(kx._vocabulary)}

        lo, hi = self.config.resolution_range or (0.01, 1.0)
        search = []
        for res in np.geomspace(lo, hi, self.config.auto_resolution_steps):
            labels = np.array(la.find_partition(
                graph, la.RBConfigurationVertexPartition, weights="weight",
                resolution_parameter=float(res), seed=self.config.random_state,
            ).membership)
            sizes = np.bincount(labels)
            labels[sizes[labels] < self.config.min_cluster_size] = -1
            n_found = len(set(labels.tolist()) - {-1})
            if n_found < 2:
                continue
            max_share = float(sizes[sizes >= self.config.min_cluster_size].max() / len(labels))
            topic_kw = kx.extract_all_topics(documents, labels, n_keywords=10, method="ctfidf")
            term_ids = [[term_index[w] for w in kws] for kws, _ in topic_kw.values()]
            coherence = float(np.mean(coherence_from_doc_term(doc_term, term_ids)))
            search.append((float(res), n_found, coherence, max_share))

        self.resolution_search_ = [(r, n, c) for r, n, c, _ in search]
        if not search:
            return self.config.resolution
        # Skip degenerate partitions dominated by one giant topic
        candidates = [x for x in search if x[3] <= self.config.auto_resolution_max_share] or search
        best = max(c for _, _, c, _ in candidates)
        threshold = best - self.config.auto_resolution_tolerance * abs(best)
        chosen = min(r for r, _, c, _ in candidates if c >= threshold)

        if self.config.verbose:
            n_at = next(n for r, n, _ in search if r == chosen)
            print(f"   > Auto resolution: {chosen:.3f} (~{n_at} topics, "
                  f"scanned {len(search)} resolutions by keyword coherence)")
        return chosen

    def _auto_resolve_topic_count(
        self,
        documents: list[str],
        metadata_graph: Any | None,
        current_n_topics: int,
    ) -> None:
        """Binary-search for a resolution that yields the target n_topics.

        Works in both directions: lowers resolution when we have too many
        topics, raises it when we have too few.  Re-runs a single-pass fit
        at the best resolution found, then refreshes all downstream state.
        """
        target = self.n_topics
        if not isinstance(target, int) or target == current_n_topics:
            return

        if self.config.verbose:
            print(f"\n   > Auto-tuning resolution for {target} topics (currently {current_n_topics})...")

        # Reuse the existing graph — it is identical to what fit() built
        # (find_optimal_resolution and fit_predict only read the graph)
        graph = self.graph_

        # Set search range based on direction
        if target > current_n_topics:
            # Need more topics → search higher resolutions
            res_range = (self.config.resolution, self.config.resolution * 20)
        else:
            # Need fewer topics → search lower resolutions
            res_range = (self.config.resolution / 1000, self.config.resolution)

        best_res = self._clusterer.find_optimal_resolution(
            graph,
            resolution_range=res_range,
            n_steps=20,
            target_n_topics=target,
            min_cluster_size=self.config.min_cluster_size,
        )

        if self.config.verbose:
            print(f"      Found resolution={best_res:.3f}")

        # Consensus clustering at the tuned resolution (edge-level consensus
        # is O(edges), so it stays cheap even for few, large clusters).
        self.graph_ = graph
        self.labels_ = self._clusterer.fit_predict(
            graph,
            min_cluster_size=self.config.min_cluster_size,
            resolution=best_res,
        )

        new_n = len(np.unique(self.labels_[self.labels_ != -1]))

        # If we overshot, merge down
        if new_n > target:
            # Temporarily mark as fitted so reduce_topics works
            was_fitted = self._is_fitted
            self._is_fitted = True
            self._extract_topic_info(documents)
            self._compute_topic_centroids()
            self.reduce_topics(target)
            self._is_fitted = was_fitted
        else:
            self._extract_topic_info(documents)
            self._compute_topic_centroids()
            self._compute_probabilities()

        final_n = len([t for t in self.topics_ if t.topic_id != -1])
        if self.config.verbose:
            print(f"      Final topic count: {final_n}")

    def _refine_embeddings(
        self,
        original_embeddings: np.ndarray,
        labels: np.ndarray,
        blend_factor: float = 0.2,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> np.ndarray:
        """
        Refine embeddings by incorporating topic context.

        Uses distance-aware blending: documents close to their topic
        centroid are pulled more strongly, while borderline documents
        are blended more conservatively to avoid misplacement.
        Outlier documents (label == -1) are left unchanged.

        Parameters
        ----------
        blend_factor : float
            Base blend strength (0 = no change, 1 = replace).
        metric : str
            "cosine" for unit-norm embeddings (re-normalized afterwards),
            "euclidean" for reduced (UMAP/PaCMAP) coordinates.
        """
        refined = original_embeddings.copy()
        unique_labels = np.unique(labels[labels != -1])

        # Compute topic centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = original_embeddings[mask].mean(axis=0)

        for label in unique_labels:
            mask = labels == label
            centroid = centroids[label]
            topic_embs = refined[mask]

            # Scale blend: core members get full blend, borderline members
            # get reduced blend
            if metric == "cosine":
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
                emb_norms = topic_embs / (np.linalg.norm(topic_embs, axis=1, keepdims=True) + 1e-10)
                cos_sim = emb_norms @ centroid_norm  # shape (n_topic_docs,)
                per_doc_scale = np.clip(cos_sim, 0.0, 1.0) ** 0.5  # sqrt for softer scaling
            else:
                dist = np.linalg.norm(topic_embs - centroid, axis=1)
                scale = np.median(dist) + 1e-10
                per_doc_scale = 1.0 / (1.0 + (dist / scale) ** 2)  # 1 at core, 0.5 at median
            per_doc_blend = blend_factor * per_doc_scale[:, np.newaxis]

            refined[mask] = (1 - per_doc_blend) * topic_embs + per_doc_blend * centroid

        if metric == "cosine":
            # Re-normalize (safe against zero-norm)
            norms = np.linalg.norm(refined, axis=1, keepdims=True)
            refined = refined / np.maximum(norms, 1e-10)

        return refined
    
    def _reduce_dimensions(self) -> None:
        """Reduce embedding dimensionality for better graph construction."""
        if self.config.verbose:
            print(f"   > Reducing dimensions to {self.config.reduced_dims}d "
                  f"({self.config.dim_reduction_method})...")

        if self.config.dim_reduction_method == "umap":
            from umap import UMAP
            self._dim_reducer = UMAP(
                n_components=self.config.reduced_dims,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric="cosine",
                random_state=self.config.random_state,
            )
        elif self.config.dim_reduction_method == "pacmap":
            from pacmap import PaCMAP
            self._dim_reducer = PaCMAP(
                n_components=self.config.reduced_dims,
                n_neighbors=self.config.umap_n_neighbors,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown dim_reduction_method: {self.config.dim_reduction_method}")

        self.reduced_embeddings_ = self._dim_reducer.fit_transform(self.embeddings_)

    def _compute_probabilities(self) -> None:
        """Compute soft topic assignment probabilities for training documents."""
        if self.config.soft_assignment_method == "graph":
            self._compute_graph_probabilities()
        else:
            self._compute_centroid_probabilities()

    def _compute_centroid_probabilities(self) -> None:
        """Centroid-based soft assignment: softmax over cosine similarity to topic centroids."""
        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
        if self.topic_embeddings_ is None or base_emb is None:
            return

        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.special import softmax

        sim_matrix = cosine_similarity(base_emb, self.topic_embeddings_)
        # Temperature scaling: higher T -> sharper peaks
        self.probabilities_ = softmax(sim_matrix * self.config.softmax_temperature, axis=1)

    def _compute_graph_probabilities(self) -> None:
        """Graph-based soft assignment: topic distribution of each document's graph neighbours."""
        if self.graph_ is None or self.labels_ is None:
            self._compute_centroid_probabilities()
            return

        from scipy.sparse import csr_matrix as _csr

        # Use same topic order as _compute_topic_centroids / _compute_centroid_probabilities
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        n_topics = len(non_outlier_topics)
        n_docs = len(self.labels_)

        # Map raw label → column index (only non-outlier topics)
        label_to_col = np.full(self.labels_.max() + 2, -1, dtype=np.intp)
        for col, tid in enumerate(non_outlier_topics):
            label_to_col[tid] = col

        # Weighted adjacency as sparse matrix (symmetric, CSR)
        adj = self.graph_.get_adjacency_sparse(attribute="weight")

        # Build one-hot topic indicator: T[doc, col] = 1 iff doc belongs to topic col
        col_indices = label_to_col[self.labels_]
        valid = col_indices >= 0
        T = _csr(
            (np.ones(valid.sum(), dtype=np.float64),
             (np.where(valid)[0], col_indices[valid])),
            shape=(n_docs, n_topics),
        )

        # proba[i, t] = sum of edge weights from doc i to all neighbours in topic t
        proba = adj.dot(T).toarray()

        # Normalize rows; fallback to uniform where row sum is zero
        row_sums = proba.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.ravel() == 0)
        row_sums[zero_rows] = 1.0          # avoid division by zero
        proba /= row_sums
        proba[zero_rows] = 1.0 / n_topics  # uniform fallback

        self.probabilities_ = proba

    def _extract_topic_info(self, documents: list[str]) -> None:
        """Extract keywords and representative documents for each topic."""
        self.topics_ = []
        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_

        # All topics in one pass (the document-term matrix is cached)
        topic_keywords = self._keyword_extractor.extract_all_topics(
            documents,
            self.labels_,
            n_keywords=self.config.n_keywords,
            include_outliers=True,
        )

        for label in np.unique(self.labels_):
            mask = self.labels_ == label
            topic_indices = np.where(mask)[0]
            keywords, scores = topic_keywords[int(label)]

            # Find representative documents (closest to centroid)
            if base_emb is not None and label != -1:
                topic_embeddings = base_emb[mask]
                centroid = topic_embeddings.mean(axis=0)
                distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
                top_indices = np.argsort(distances)[:self.config.n_representative_docs]
                representative_docs = [int(topic_indices[i]) for i in top_indices]
            else:
                representative_docs = [int(i) for i in topic_indices[:self.config.n_representative_docs]]

            topic_info = TopicInfo(
                topic_id=int(label),
                size=int(mask.sum()),
                keywords=keywords,
                keyword_scores=scores,
                representative_docs=representative_docs,
                label=None,
                description=None,
            )
            self.topics_.append(topic_info)

        # Sort by size (excluding outliers)
        self.topics_ = sorted(
            self.topics_,
            key=lambda t: (t.topic_id == -1, -t.size)
        )

    def _compute_topic_centroids(self) -> None:
        """Compute centroid embeddings for each topic.

        Uses original (unrefined) embeddings so that ``transform()`` on new
        documents operates in the same embedding space as the centroids.
        """
        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
        if base_emb is None:
            return

        unique_labels = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        self.topic_embeddings_ = np.zeros((len(unique_labels), base_emb.shape[1]))

        topic_lookup = {t.topic_id: t for t in self.topics_}

        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            self.topic_embeddings_[i] = base_emb[mask].mean(axis=0)
            if label in topic_lookup:
                topic_lookup[label].centroid = self.topic_embeddings_[i]
    
    def get_document_topics(
        self,
        doc_idx: int,
        top_n: int = 3,
        method: Literal["centroid", "graph"] | None = None,
    ) -> list[tuple[int, float]]:
        """Return top-N topics with probabilities for a single document.

        Parameters
        ----------
        doc_idx : int
            Index of the document.
        top_n : int
            Number of top topics to return.
        method : str, optional
            ``"centroid"`` or ``"graph"``.  If *None*, uses
            ``self.config.soft_assignment_method``.

        Returns
        -------
        topics : list[tuple[int, float]]
            List of ``(topic_id, probability)`` sorted descending.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        method = method or self.config.soft_assignment_method
        # Use same topic order as _compute_topic_centroids (matches topic_embeddings_ columns)
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]

        if method == "graph" and self.graph_ is not None:
            topic_idx = {tid: i for i, tid in enumerate(non_outlier_topics)}
            proba = np.zeros(len(non_outlier_topics))
            neighbors = self.graph_.neighbors(doc_idx)
            for nb in neighbors:
                eid = self.graph_.get_eid(doc_idx, nb)
                w = self.graph_.es[eid]["weight"]
                lab = self.labels_[nb]
                if lab != -1 and lab in topic_idx:
                    proba[topic_idx[lab]] += w
            s = proba.sum()
            if s > 0:
                proba /= s
            else:
                proba[:] = 1.0 / len(non_outlier_topics)
        else:
            # Centroid-based
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.special import softmax

            base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
            sim = cosine_similarity(base_emb[doc_idx:doc_idx+1], self.topic_embeddings_)[0]
            proba = softmax(sim * self.config.softmax_temperature)

        ranked = np.argsort(proba)[::-1][:top_n]
        return [(non_outlier_topics[i], float(proba[i])) for i in ranked]

    def topic_overlap_matrix(self, threshold: float = 0.1) -> pd.DataFrame:
        """Compute a topic co-occurrence matrix from soft assignments.

        For each document, topics whose probability exceeds *threshold* are
        considered "active".  The matrix counts how often each pair of topics
        co-occurs across documents.

        Parameters
        ----------
        threshold : float
            Minimum probability for a topic to count as active.

        Returns
        -------
        overlap : pd.DataFrame
            Symmetric ``(n_topics, n_topics)`` DataFrame of co-occurrence counts.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.probabilities_ is None:
            self._compute_probabilities()

        # Use same topic order as _compute_topic_centroids (matches probabilities_ columns)
        non_outlier_topics = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        n_topics = len(non_outlier_topics)
        active = (self.probabilities_ >= threshold).astype(np.int64)
        overlap = active.T @ active

        labels = [f"Topic {tid}" for tid in non_outlier_topics]
        return pd.DataFrame(overlap, index=labels, columns=labels)

    def visualize_overlap(self, threshold: float = 0.1, **kwargs):
        """Visualize the topic overlap matrix as a heatmap.

        Parameters
        ----------
        threshold : float
            Minimum probability for a topic to count as active.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        from tritopic.visualization.plotter import plot_topic_overlap

        overlap = self.topic_overlap_matrix(threshold)
        topics = [t for t in self.topics_ if t.topic_id != -1]
        return plot_topic_overlap(overlap, topics, **kwargs)

    # ------------------------------------------------------------------
    # Hierarchical Topics
    # ------------------------------------------------------------------

    def build_hierarchy(
        self,
        resolution_levels: list[float] | None = None,
        n_levels: int = 3,
    ) -> TopicHierarchy:
        """Build a multi-resolution topic hierarchy.

        Re-uses the existing graph and clusters it at multiple resolution
        levels.  Coarse levels (low resolution) give broad themes; fine
        levels (high resolution) give specific sub-topics.  Levels are
        linked by majority-vote: each fine-grained node is assigned to
        the coarse-grained node that contains the majority of its
        documents.

        Parameters
        ----------
        resolution_levels : list[float], optional
            Explicit Leiden resolution values from coarse to fine.  If
            *None*, auto-generates *n_levels* values geometrically
            spaced between ``resolution_ / 4`` and ``resolution_ * 4``
            (``resolution_`` is the resolution chosen by ``fit()``).
        n_levels : int
            Number of levels when *resolution_levels* is *None*.

        Returns
        -------
        hierarchy : TopicHierarchy
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        import leidenalg as la

        if resolution_levels is None:
            base = self.resolution_  # resolution actually used by fit()
            resolution_levels = list(np.geomspace(base / 4, base * 4, n_levels))

        resolution_levels = sorted(resolution_levels)  # coarse → fine
        graph = self.graph_

        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_

        all_level_nodes: list[list[TopicNode]] = []

        for level_idx, res in enumerate(resolution_levels):
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=self.config.random_state,
            )
            level_labels = np.array(partition.membership)
            unique_ids = sorted(set(level_labels))

            # Keywords for all clusters of this level in one pass (reuses the
            # cached document-term matrix instead of re-tokenizing the corpus
            # for every cluster)
            level_keywords = self._keyword_extractor.extract_all_topics(
                self.documents_, level_labels, n_keywords=self.config.n_keywords,
            )

            level_nodes: list[TopicNode] = []
            for tid in unique_ids:
                doc_idx = np.where(level_labels == tid)[0]
                centroid = base_emb[doc_idx].mean(axis=0) if base_emb is not None else None
                keywords, scores = level_keywords[int(tid)]

                node = TopicNode(
                    node_id=f"L{level_idx}_{tid}",
                    level=level_idx,
                    topic_id=tid,
                    size=len(doc_idx),
                    keywords=keywords,
                    keyword_scores=scores,
                    doc_indices=doc_idx,
                    centroid=centroid,
                )
                level_nodes.append(node)

            all_level_nodes.append(level_nodes)

        # Link levels via majority-vote
        for lvl in range(1, len(all_level_nodes)):
            parent_nodes = all_level_nodes[lvl - 1]
            child_nodes = all_level_nodes[lvl]

            # Build parent lookup: doc_idx → parent node
            parent_of_doc: dict[int, TopicNode] = {}
            for pnode in parent_nodes:
                for di in pnode.doc_indices:
                    parent_of_doc[di] = pnode

            for cnode in child_nodes:
                # Majority vote: which parent has the most overlap?
                votes: dict[str, int] = {}
                for di in cnode.doc_indices:
                    pn = parent_of_doc.get(di)
                    if pn is not None:
                        votes[pn.node_id] = votes.get(pn.node_id, 0) + 1

                if votes:
                    best_parent_id = max(votes, key=votes.get)
                    for pnode in parent_nodes:
                        if pnode.node_id == best_parent_id:
                            cnode.parent = pnode
                            pnode.children.append(cnode)
                            break

        hierarchy = TopicHierarchy(
            roots=all_level_nodes[0],
            levels=all_level_nodes,
            resolution_levels=resolution_levels,
        )
        self.hierarchy_ = hierarchy

        if self.config.verbose:
            sizes = [len(lvl) for lvl in all_level_nodes]
            print(f"[Hierarchy] Built {len(sizes)} levels: {sizes} topics")

        return hierarchy

    def divide(
        self,
        topic_id: int,
        n_subtopics: int = 2,
    ) -> list[TopicInfo]:
        """Split a single topic into *n_subtopics* sub-topics.

        Extracts the subgraph for the given topic and runs Leiden on it
        at a higher resolution to discover finer sub-communities.

        Parameters
        ----------
        topic_id : int
            Topic to divide.
        n_subtopics : int
            Target number of sub-topics.

        Returns
        -------
        subtopics : list[TopicInfo]
            New topic info objects for the sub-topics.  The labels in
            ``self.labels_`` are updated in-place; the original topic is
            replaced by the new sub-topics.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        import leidenalg as la

        mask = self.labels_ == topic_id
        if not np.any(mask):
            raise ValueError(f"Topic {topic_id} not found.")

        doc_indices = np.where(mask)[0]
        subgraph = self.graph_.subgraph(doc_indices.tolist())

        # Find resolution that yields ~n_subtopics
        sub_clusterer = ConsensusLeiden(
            resolution=self.config.resolution,
            n_runs=self.config.n_consensus_runs,
            random_state=self.config.random_state,
        )
        best_res = sub_clusterer.find_optimal_resolution(
            subgraph,
            # Search both directions: on a subgraph the current resolution
            # may already give more than n_subtopics clusters.
            resolution_range=(self.config.resolution / 1000, self.config.resolution * 20),
            n_steps=20,
            target_n_topics=n_subtopics,
            min_cluster_size=max(2, self.config.min_cluster_size // 2),
        )

        sub_labels = sub_clusterer.fit_predict(
            subgraph, min_cluster_size=max(2, self.config.min_cluster_size // 2),
            resolution=best_res,
        )

        # Map sub-labels into global label space
        existing_max = int(self.labels_.max())
        new_ids = []
        for sub_id in sorted(set(sub_labels[sub_labels != -1])):
            new_label = existing_max + 1 + int(sub_id)
            self.labels_[doc_indices[sub_labels == sub_id]] = new_label
            new_ids.append(new_label)

        # Docs that became outliers in the subgraph keep original topic_id
        outlier_mask = sub_labels == -1
        if np.any(outlier_mask):
            self.labels_[doc_indices[outlier_mask]] = topic_id

        # Refresh topics list and centroids (keywords, representative docs
        # and centroids of the new sub-topics come from here)
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()
        new_topics = [self.get_topic(tid) for tid in new_ids]

        if self.config.verbose:
            print(f"Divided topic {topic_id} into {len(new_topics)} sub-topics")

        return new_topics

    def visualize_hierarchy_tree(self, **kwargs):
        """Visualize the topic hierarchy as a tree diagram.

        Requires :meth:`build_hierarchy` to be called first.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        from tritopic.visualization.plotter import plot_hierarchy_tree

        if self.hierarchy_ is None:
            raise ValueError("No hierarchy built. Call build_hierarchy() first.")

        return plot_hierarchy_tree(self.hierarchy_, **kwargs)

    def fit_transform(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Fit the model and return topic assignments.

        Parameters
        ----------
        documents : list[str]
            List of document texts.
        embeddings : np.ndarray, optional
            Pre-computed embeddings.
        metadata : pd.DataFrame, optional
            Document metadata.
            
        Returns
        -------
        labels : np.ndarray
            Topic assignment for each document. -1 indicates outlier.
        """
        self.fit(documents, embeddings, metadata)
        return self.labels_
    
    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Assign topics to new documents.

        Parameters
        ----------
        documents : list[str]
            New documents to classify.

        Returns
        -------
        labels : np.ndarray
            Topic assignments.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity

        new_embeddings = self._embedding_engine.encode(documents)

        non_outlier_topics = [t for t in self.topics_ if t.topic_id != -1]
        topic_ids = np.array([t.topic_id for t in non_outlier_topics])

        sim_matrix = cosine_similarity(new_embeddings, self.topic_embeddings_)
        nearest_idx = np.argmax(sim_matrix, axis=1)
        max_sim = sim_matrix[np.arange(len(documents)), nearest_idx]

        labels = topic_ids[nearest_idx]
        labels[max_sim < self.config.outlier_threshold] = -1

        return labels

    def transform_proba(self, documents: list[str]) -> np.ndarray:
        """
        Get soft topic assignment probabilities for new documents.

        Parameters
        ----------
        documents : list[str]
            New documents to classify.

        Returns
        -------
        probabilities : np.ndarray
            Shape (n_docs, n_topics) probability matrix. Rows sum to ~1.0.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.special import softmax

        new_embeddings = self._embedding_engine.encode(documents)
        sim_matrix = cosine_similarity(new_embeddings, self.topic_embeddings_)
        return softmax(sim_matrix * self.config.softmax_temperature, axis=1)

    def reduce_outliers(
        self,
        strategy: Literal["embeddings", "neighbors"] = "embeddings",
        threshold: float | None = None,
    ) -> "TriTopic":
        """
        Reassign outlier documents to the nearest topic.

        Parameters
        ----------
        strategy : str
            "embeddings" — assign each outlier to the most similar topic centroid
            (if similarity > threshold).
            "neighbors" — assign each outlier by majority vote of its k nearest
            non-outlier neighbors in embedding space.
        threshold : float, optional
            Minimum cosine similarity for assignment (embeddings strategy only).
            Defaults to ``self.config.outlier_threshold``.

        Returns
        -------
        self : TriTopic
            Updated model (labels_, topics_, topic_embeddings_, probabilities_).
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        outlier_mask = self.labels_ == -1
        if not np.any(outlier_mask):
            if self.config.verbose:
                print("No outliers to reduce.")
            return self

        outlier_indices = np.where(outlier_mask)[0]

        if self.config.verbose:
            print(f"Reducing {len(outlier_indices)} outliers (strategy={strategy})...")

        if strategy == "embeddings":
            from sklearn.metrics.pairwise import cosine_similarity

            thresh = threshold if threshold is not None else self.config.outlier_threshold
            non_outlier_topics = [t for t in self.topics_ if t.topic_id != -1]
            # Same space as topic_embeddings_ (original, unrefined)
            base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
            sim_matrix = cosine_similarity(
                base_emb[outlier_indices], self.topic_embeddings_
            )

            for local_idx, global_idx in enumerate(outlier_indices):
                best_topic_idx = int(np.argmax(sim_matrix[local_idx]))
                best_sim = sim_matrix[local_idx, best_topic_idx]
                if best_sim >= thresh:
                    self.labels_[global_idx] = non_outlier_topics[best_topic_idx].topic_id

        elif strategy == "neighbors":
            from sklearn.neighbors import NearestNeighbors

            base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
            non_outlier_mask = ~outlier_mask
            non_outlier_indices = np.where(non_outlier_mask)[0]
            non_outlier_embeddings = base_emb[non_outlier_mask]

            nn = NearestNeighbors(
                n_neighbors=min(self.config.n_neighbors, len(non_outlier_indices)),
                metric="cosine",
            )
            nn.fit(non_outlier_embeddings)
            _, neighbor_idx = nn.kneighbors(base_emb[outlier_indices])

            for local_idx, global_idx in enumerate(outlier_indices):
                neighbor_global = non_outlier_indices[neighbor_idx[local_idx]]
                neighbor_labels = self.labels_[neighbor_global]
                # Majority vote
                values, counts = np.unique(neighbor_labels, return_counts=True)
                self.labels_[global_idx] = values[np.argmax(counts)]
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'embeddings' or 'neighbors'.")

        # Refresh downstream state
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            remaining = int(np.sum(self.labels_ == -1))
            print(f"   Outliers remaining: {remaining}")

        return self

    def reduce_topics(self, n_topics: int) -> "TriTopic":
        """
        Iteratively merge the two most similar topics until *n_topics* remain.

        Parameters
        ----------
        n_topics : int
            Target number of non-outlier topics.

        Returns
        -------
        self : TriTopic
            Updated model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        base_emb = self.original_embeddings_ if self.original_embeddings_ is not None else self.embeddings_
        non_outlier_ids = [t.topic_id for t in self.topics_ if t.topic_id != -1]
        current_count = len(non_outlier_ids)

        if n_topics >= current_count:
            if self.config.verbose:
                print(f"Already at {current_count} topics (requested {n_topics}).")
            return self

        if self.config.verbose:
            print(f"Reducing from {current_count} to {n_topics} topics...")

        while current_count > n_topics:
            # Recompute centroids list aligned with current non-outlier ids
            non_outlier_ids = sorted(set(self.labels_[self.labels_ != -1]))
            sizes = np.array([
                int(np.sum(self.labels_ == tid)) for tid in non_outlier_ids
            ])
            centroids = np.array([
                base_emb[self.labels_ == tid].mean(axis=0)
                for tid in non_outlier_ids
            ])
            sim = cos_sim(centroids)

            # Size-aware merge scoring: prefer merging smaller topics.
            # Penalty = (min_size / max_size) ** 0.3 (mild): small-small -> 1,
            # small-large -> small.
            size_factor = (
                np.minimum.outer(sizes, sizes) / np.maximum.outer(sizes, sizes)
            ) ** 0.3
            sim = sim * size_factor
            np.fill_diagonal(sim, -np.inf)

            # Find best pair to merge
            flat_idx = int(np.argmax(sim))
            i, j = divmod(flat_idx, len(non_outlier_ids))
            merge_from = non_outlier_ids[j]
            merge_into = non_outlier_ids[i]
            # Keep the larger topic's id
            if sizes[j] > sizes[i]:
                merge_into, merge_from = merge_from, merge_into
            # Relabel
            self.labels_[self.labels_ == merge_from] = merge_into
            current_count -= 1

        # Refresh downstream state
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            final = len([t for t in self.topics_ if t.topic_id != -1])
            print(f"   Now have {final} topics.")

        return self

    def merge_topics(self, topics_to_merge: list[int]) -> "TriTopic":
        """
        Merge the specified topic IDs into one topic.

        The largest topic's ID is kept.

        Parameters
        ----------
        topics_to_merge : list[int]
            Topic IDs to merge together.

        Returns
        -------
        self : TriTopic
            Updated model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if len(topics_to_merge) < 2:
            raise ValueError("Need at least 2 topic IDs to merge.")

        # Determine which topic to keep (largest)
        sizes = {tid: int(np.sum(self.labels_ == tid)) for tid in topics_to_merge}
        keep_id = max(sizes, key=sizes.get)

        for tid in topics_to_merge:
            if tid != keep_id:
                self.labels_[self.labels_ == tid] = keep_id

        # Refresh downstream state
        self._extract_topic_info(self.documents_)
        self._compute_topic_centroids()
        self._compute_probabilities()

        if self.config.verbose:
            print(f"Merged topics {topics_to_merge} -> {keep_id}")

        return self

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get a DataFrame with topic information.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: Topic, Size, Keywords, Label, Coherence
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        data = []
        for topic in self.topics_:
            data.append({
                "Topic": topic.topic_id,
                "Size": topic.size,
                "Keywords": ", ".join(topic.keywords[:5]),
                "All_Keywords": topic.keywords,
                "Keyword_Scores": topic.keyword_scores,
                "Label": topic.label or f"Topic {topic.topic_id}",
                "Description": topic.description,
                "Representative_Docs": topic.representative_docs,
                "Coherence": topic.coherence,
            })
        
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> TopicInfo | None:
        """Get information about a specific topic."""
        for topic in self.topics_:
            if topic.topic_id == topic_id:
                return topic
        return None
    
    def get_representative_docs(
        self,
        topic_id: int,
        n_docs: int = 5,
    ) -> list[tuple[int, str]]:
        """
        Get representative documents for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic ID.
        n_docs : int
            Number of documents to return.
            
        Returns
        -------
        docs : list[tuple[int, str]]
            List of (index, document_text) tuples.
        """
        if not self._is_fitted or self.documents_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topic = self.get_topic(topic_id)
        if topic is None:
            raise ValueError(f"Topic {topic_id} not found.")
        
        indices = topic.representative_docs[:n_docs]
        return [(idx, self.documents_[idx]) for idx in indices]
    
    def generate_labels(
        self,
        labeler: "LLMLabeler",
        topics: list[int] | None = None,
    ) -> None:
        """
        Generate labels for topics using an LLM.
        
        Parameters
        ----------
        labeler : LLMLabeler
            Configured LLM labeler instance.
        topics : list[int], optional
            Specific topics to label. If None, labels all.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        target_topics = topics or [t.topic_id for t in self.topics_ if t.topic_id != -1]
        
        for topic_id in tqdm(target_topics, desc="Generating labels", disable=not self.config.verbose):
            topic = self.get_topic(topic_id)
            if topic is None:
                continue
            
            # Get representative docs
            rep_docs = self.get_representative_docs(topic_id, n_docs=5)
            doc_texts = [doc for _, doc in rep_docs]
            
            # Generate label
            label, description = labeler.generate_label(
                keywords=topic.keywords,
                representative_docs=doc_texts,
            )
            
            topic.label = label
            topic.description = description
    
    def visualize(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        color_by: Literal["topic", "custom"] = "topic",
        custom_labels: list[str] | None = None,
        show_outliers: bool = True,
        interactive: bool = True,
        **kwargs,
    ):
        """
        Visualize topics in 2D.
        
        Parameters
        ----------
        method : str
            Dimensionality reduction method. "umap" or "pacmap".
        color_by : str
            How to color points. "topic" uses topic assignments.
        custom_labels : list[str], optional
            Custom labels for hover text.
        show_outliers : bool
            Whether to show outlier documents.
        interactive : bool
            If True, returns interactive Plotly figure.
        **kwargs
            Additional arguments passed to the visualizer.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive visualization.
        """
        from tritopic.visualization.plotter import TopicVisualizer
        
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        visualizer = TopicVisualizer(method=method)
        
        return visualizer.plot_documents(
            embeddings=self.embeddings_,
            labels=self.labels_,
            documents=self.documents_,
            topics=self.topics_,
            show_outliers=show_outliers,
            interactive=interactive,
            **kwargs,
        )
    
    def visualize_hierarchy(self, **kwargs):
        """Visualize topic hierarchy as a dendrogram."""
        from tritopic.visualization.plotter import TopicVisualizer
        
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        visualizer = TopicVisualizer()
        return visualizer.plot_hierarchy(
            topic_embeddings=self.topic_embeddings_,
            topics=self.topics_,
            **kwargs,
        )
    
    def visualize_topics(self, **kwargs):
        """Visualize topics as a heatmap or bar chart."""
        from tritopic.visualization.plotter import TopicVisualizer
        
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        visualizer = TopicVisualizer()
        return visualizer.plot_topics(
            topics=self.topics_,
            **kwargs,
        )
    
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate topic model quality.
        
        Returns
        -------
        metrics : dict
            Dictionary with coherence, diversity, and stability scores.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # NPMI coherence against the whole corpus as reference.  (Using only
        # the topic's own documents as reference inflates the score: every
        # keyword is frequent there by construction.)
        real_topics = [t for t in self.topics_ if t.topic_id != -1]
        coherences = compute_coherence_batch(
            [t.keywords for t in real_topics], self.documents_,
            language=self.config.language,
        )
        for topic, coh in zip(real_topics, coherences):
            topic.coherence = coh
        
        # Compute diversity
        all_keywords = [kw for t in self.topics_ if t.topic_id != -1 for kw in t.keywords]
        diversity = compute_diversity(all_keywords, n_topics=len(coherences))
        
        # Get stability from consensus clustering
        stability = self._clusterer.stability_score_ if hasattr(self._clusterer, 'stability_score_') else None
        
        metrics = {
            "coherence_mean": float(np.mean(coherences)) if coherences else 0.0,
            "coherence_std": float(np.std(coherences)) if coherences else 0.0,
            "diversity": diversity,
            "stability": stability,
            "n_topics": len([t for t in self.topics_ if t.topic_id != -1]),
            "outlier_ratio": float(np.mean(self.labels_ == -1)) if self.labels_ is not None else 0.0,
        }
        
        if self.config.verbose:
            print("\n[Metrics] Evaluation:")
            print(f"   Coherence (mean): {metrics['coherence_mean']:.4f}")
            print(f"   Diversity: {metrics['diversity']:.4f}")
            if stability:
                print(f"   Stability: {stability:.4f}")
            print(f"   Outlier ratio: {metrics['outlier_ratio']:.2%}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle

        state = {
            "config": self.config,
            "n_topics": self.n_topics,
            "topics_": self.topics_,
            "labels_": self.labels_,
            "embeddings_": self.embeddings_,
            "original_embeddings_": self.original_embeddings_,
            "reduced_embeddings_": self.reduced_embeddings_,
            "probabilities_": self.probabilities_,
            "lexical_matrix_": self.lexical_matrix_,
            "topic_embeddings_": self.topic_embeddings_,
            "documents_": self.documents_,
            "hierarchy_": self.hierarchy_,
            "_is_fitted": self._is_fitted,
            "_iteration_history": self._iteration_history,
            "resolution_": self.resolution_,
            "resolution_search_": self.resolution_search_,
            "_dim_reducer": self._dim_reducer,
            "_keyword_extractor_state": {
                "vectorizer": self._keyword_extractor._vectorizer,
                "vocabulary": self._keyword_extractor._vocabulary,
                "idf": getattr(self._keyword_extractor, "_idf", None),
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        if self.config.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TriTopic":
        """Load model from disk."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        # Backward compat: ensure new config fields exist for models saved before v2.2
        if not hasattr(config, "language"):
            config.language = "english"
        if not hasattr(config, "soft_assignment_method"):
            config.soft_assignment_method = "centroid"

        model = cls(config=config)
        model.n_topics = state.get("n_topics", "auto")
        model.topics_ = state["topics_"]
        model.labels_ = state["labels_"]
        model.embeddings_ = state["embeddings_"]
        model.original_embeddings_ = state.get("original_embeddings_")
        model.reduced_embeddings_ = state.get("reduced_embeddings_")
        model.probabilities_ = state.get("probabilities_")
        model.lexical_matrix_ = state.get("lexical_matrix_")
        model.topic_embeddings_ = state["topic_embeddings_"]
        model.documents_ = state["documents_"]
        model.hierarchy_ = state.get("hierarchy_")
        model._is_fitted = state["_is_fitted"]
        model._iteration_history = state["_iteration_history"]
        model.resolution_ = state.get("resolution_", config.resolution)
        model.resolution_search_ = state.get("resolution_search_", [])
        model._dim_reducer = state.get("_dim_reducer")

        # Restore keyword extractor state
        kw_state = state.get("_keyword_extractor_state")
        if kw_state:
            model._keyword_extractor._vectorizer = kw_state.get("vectorizer")
            model._keyword_extractor._vocabulary = kw_state.get("vocabulary")
            model._keyword_extractor._idf = kw_state.get("idf")

        return model
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_topics = len([t for t in self.topics_ if t.topic_id != -1]) if self._is_fitted else "?"
        return f"TriTopic(n_topics={n_topics}, status={status})"
