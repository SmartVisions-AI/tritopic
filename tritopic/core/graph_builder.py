"""
Graph Builder for TriTopic
============================

Constructs similarity graphs using multiple strategies:
- Mutual kNN: Only keep edges where both nodes are in each other's neighborhood
- SNN (Shared Nearest Neighbors): Weight edges by number of shared neighbors
- Multi-view fusion: Combine semantic, lexical, and metadata graphs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MetadataFeatures:
    """Encoded document metadata (see ``GraphBuilder.build_metadata_graph``)."""

    categorical: np.ndarray  # (n_docs, n_cat) integer codes, -1 = missing
    numerical: np.ndarray    # (n_docs, n_num) values in [0, 1], NaN = missing


class GraphBuilder:
    """
    Build similarity graphs for topic modeling.
    
    Supports multiple graph construction strategies for robust clustering.
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for kNN graph. Default: 15
    metric : str
        Distance metric. Default: "cosine"
    graph_type : str
        Type of graph: "knn", "mutual_knn", "snn", or "hybrid"
    snn_weight : float
        Weight for SNN edges in hybrid mode. Default: 0.5
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "cosine",
        graph_type: Literal["knn", "mutual_knn", "snn", "hybrid"] = "hybrid",
        snn_weight: float = 0.5,
        language: str = "english",
    ):
        from tritopic.utils.stopwords import TOKEN_PATTERN, get_stopwords

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.graph_type = graph_type
        self.snn_weight = snn_weight
        self.language = language

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=get_stopwords(language),
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # log(1+tf) dampens common term dominance
        )
        # (tfidf_matrix, k, lexical_adj): the lexical kNN graph only depends on
        # the TF-IDF matrix, so it is built once per fit.
        self._lexical_cache: tuple | None = None
    
    def _compute_knn(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute kNN and return (distances, indices, similarities).

        Shared helper so hybrid graphs avoid duplicate kNN computation.
        Uses FAISS when available (5-10x faster), falls back to sklearn.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = embeddings.shape[0]
        k_actual = min(k + 1, n_samples)

        try:
            import faiss
            emb = np.ascontiguousarray(embeddings, dtype=np.float32)
            if self.metric == "cosine":
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1
                emb = emb / norms
                index = faiss.IndexFlatIP(emb.shape[1])
                index.add(emb)
                sims, indices = index.search(emb, k_actual)
                distances = 1 - sims
            else:
                index = faiss.IndexFlatL2(emb.shape[1])
                index.add(emb)
                sq_distances, indices = index.search(emb, k_actual)
                distances = np.sqrt(np.maximum(sq_distances, 0))  # FAISS returns squared L2
        except ImportError:
            nn = NearestNeighbors(
                n_neighbors=k_actual,
                metric=self.metric,
                algorithm="auto",
                n_jobs=-1,
            )
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)

        if self.metric == "cosine":
            similarities = 1 - distances
        else:
            # Self-tuning Gaussian kernel (Zelnik-Manor & Perona): sigma_i is
            # the distance to the k-th neighbour, so dense and sparse regions
            # get comparable edge weights.  1/(1+d) is scale-dependent and
            # nearly flat on UMAP output.
            sigma = np.maximum(distances[:, -1], 1e-10)
            similarities = np.exp(
                -(distances ** 2) / (sigma[:, None] * sigma[indices])
            )

        return distances, indices, similarities

    def build_knn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build a basic kNN graph (vectorized).

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings of shape (n_docs, n_dims).
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix with cosine similarity weights.
        """
        n_samples = embeddings.shape[0]
        _, indices, similarities = self._compute_knn(embeddings, n_neighbors)

        # Vectorized construction: exclude self-loops (col 0 is self)
        rows = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols = indices[:, 1:].ravel()
        data = similarities[:, 1:].ravel()

        adjacency = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        return adjacency
    
    def build_mutual_knn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
        _precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> csr_matrix:
        """
        Build a mutual kNN graph (vectorized).

        Edge (i, j) exists only if i is in j's neighbors AND j is in i's neighbors.
        This removes "one-way" connections that often represent noise.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.
        _precomputed : tuple, optional
            Pre-computed (distances, indices, similarities) from _compute_knn.

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix.
        """
        n_samples = embeddings.shape[0]

        if _precomputed is not None:
            _, indices, similarities = _precomputed
        else:
            _, indices, similarities = self._compute_knn(embeddings, n_neighbors)

        # Build directed kNN adjacency (excluding self = col 0)
        rows_dir = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols_dir = indices[:, 1:].ravel()
        data_dir = similarities[:, 1:].ravel()

        knn_adj = csr_matrix(
            (data_dir, (rows_dir, cols_dir)), shape=(n_samples, n_samples)
        )

        # Mutual = element-wise minimum of knn_adj and knn_adj.T
        # (non-zero only where both directions exist)
        knn_T = knn_adj.T.tocsr()
        mutual = knn_adj.minimum(knn_T)  # keeps mutual edges, min similarity

        # Average forward and reverse similarities for mutual edges
        mutual_avg = (knn_adj + knn_T).multiply(mutual > 0) / 2

        return mutual_avg.tocsr()
    
    def build_snn_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
        _precomputed_indices: np.ndarray | None = None,
    ) -> csr_matrix:
        """
        Build a Shared Nearest Neighbors (SNN) graph (vectorized).

        Uses sparse matrix multiplication B·B^T to compute shared neighbor
        counts entirely in C-level BLAS — orders of magnitude faster than
        Python-level set intersection loops.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.
        _precomputed_indices : np.ndarray, optional
            Pre-computed kNN indices (internal use by hybrid graph).

        Returns
        -------
        adjacency : csr_matrix
            Sparse adjacency matrix with SNN weights.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = embeddings.shape[0]

        if _precomputed_indices is not None:
            indices = _precomputed_indices
        else:
            nn = NearestNeighbors(
                n_neighbors=min(k + 1, n_samples),
                metric=self.metric,
                algorithm="auto",
            )
            nn.fit(embeddings)
            _, indices = nn.kneighbors(embeddings)

        # Build binary kNN indicator matrix (exclude self at col 0)
        neighbor_indices = indices[:, 1:]  # (n_samples, k)
        rows = np.repeat(np.arange(n_samples), neighbor_indices.shape[1])
        cols = neighbor_indices.ravel()
        data = np.ones(len(rows), dtype=np.float32)
        B = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

        # Shared neighbor counts via sparse matrix multiplication (C-level)
        snn = B.dot(B.T)

        # Keep only edges where at least one kNN direction exists
        knn_union = B + B.T  # non-zero where i->j or j->i in kNN
        snn = snn.multiply(knn_union > 0)

        # Normalize by k
        snn.data = snn.data / k

        # Remove self-loops
        snn.setdiag(0)
        snn.eliminate_zeros()

        return snn.tocsr()
    
    def build_hybrid_graph(
        self,
        embeddings: np.ndarray,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build a hybrid graph combining mutual kNN and SNN.

        Computes kNN once and shares the result between both sub-graphs.

        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Combined adjacency matrix.
        """
        # Compute kNN ONCE and share
        precomputed = self._compute_knn(embeddings, n_neighbors)
        _, indices, _ = precomputed

        mutual_adj = self.build_mutual_knn_graph(
            embeddings, n_neighbors, _precomputed=precomputed
        )
        snn_adj = self.build_snn_graph(
            embeddings, n_neighbors, _precomputed_indices=indices
        )

        # Normalize both
        mutual_max = mutual_adj.max() if mutual_adj.nnz > 0 else 1
        snn_max = snn_adj.max() if snn_adj.nnz > 0 else 1

        if mutual_max > 0:
            mutual_adj = mutual_adj / mutual_max
        if snn_max > 0:
            snn_adj = snn_adj / snn_max

        # Combine
        combined = (1 - self.snn_weight) * mutual_adj + self.snn_weight * snn_adj

        return combined.tocsr()
    
    def build_lexical_matrix(
        self,
        documents: list[str],
    ) -> csr_matrix:
        """
        Build TF-IDF matrix for lexical similarity.
        
        Parameters
        ----------
        documents : list[str]
            Document texts.
            
        Returns
        -------
        tfidf_matrix : csr_matrix
            TF-IDF sparse matrix.
        """
        tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents)
        return tfidf_matrix

    def build_lexical_matrix_from_counts(
        self,
        doc_term: csr_matrix,
        max_features: int = 10000,
    ) -> csr_matrix:
        """
        Build the TF-IDF matrix from an existing document-term count matrix.

        Equivalent to :meth:`build_lexical_matrix` (top ``max_features``
        terms by corpus frequency, sublinear TF, smoothed IDF, L2 norm) but
        avoids tokenizing the corpus a second time -- the keyword extractor
        already holds the counts.  On long documents tokenization dominates
        the runtime.
        """
        from sklearn.feature_extraction.text import TfidfTransformer

        doc_term = csr_matrix(doc_term)
        if doc_term.shape[1] > max_features:
            freq = np.asarray(doc_term.sum(axis=0)).ravel()
            keep = np.sort(np.argpartition(-freq, max_features - 1)[:max_features])
            doc_term = doc_term[:, keep]
        return TfidfTransformer(sublinear_tf=True).fit_transform(doc_term).tocsr()
    
    def build_lexical_graph(
        self,
        tfidf_matrix: csr_matrix,
        n_neighbors: int | None = None,
    ) -> csr_matrix:
        """
        Build lexical similarity graph from TF-IDF (vectorized mutual kNN).

        Parameters
        ----------
        tfidf_matrix : csr_matrix
            TF-IDF matrix.
        n_neighbors : int, optional
            Override default n_neighbors.

        Returns
        -------
        adjacency : csr_matrix
            Lexical similarity adjacency matrix.
        """
        k = n_neighbors or self.n_neighbors
        n_samples = tfidf_matrix.shape[0]

        cache = self._lexical_cache
        if cache is not None and cache[0] is tfidf_matrix and cache[1] == k:
            return cache[2]

        nn = NearestNeighbors(
            n_neighbors=min(k + 1, n_samples),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(tfidf_matrix)
        distances, indices = nn.kneighbors(tfidf_matrix)

        similarities = 1 - distances

        # Vectorized mutual kNN
        rows_dir = np.repeat(np.arange(n_samples), indices.shape[1] - 1)
        cols_dir = indices[:, 1:].ravel()
        data_dir = similarities[:, 1:].ravel()

        knn_adj = csr_matrix(
            (data_dir, (rows_dir, cols_dir)), shape=(n_samples, n_samples)
        )
        knn_T = knn_adj.T.tocsr()

        # Mutual edges: average similarities where both directions exist
        mutual = knn_adj.minimum(knn_T)
        mutual_avg = (knn_adj + knn_T).multiply(mutual > 0) / 2

        mutual_avg = mutual_avg.tocsr()
        self._lexical_cache = (tfidf_matrix, k, mutual_avg)
        return mutual_avg
    
    def build_metadata_graph(
        self,
        metadata: "pd.DataFrame",
    ) -> "MetadataFeatures":
        """
        Encode metadata for the metadata view.

        Categorical columns (strings, categories, booleans) are matched
        exactly; numerical/datetime columns are min-max normalized and count
        as similar when they differ by less than 0.2.

        The metadata view is applied as a *reweighting of existing edges*
        (semantic / lexical) in :meth:`build_multiview_graph`.  Materializing
        all same-category pairs would create O(n^2 / n_categories) edges --
        e.g. 12M edges for 6k documents with 3 sources -- which both makes
        Leiden very slow and lets the metadata cliques dominate the topics.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata DataFrame aligned with the documents.

        Returns
        -------
        features : MetadataFeatures
            Encoded metadata, consumed by ``build_multiview_graph``.
        """
        import pandas as pd

        categorical, numerical = [], []
        for col in metadata.columns:
            s = metadata[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                s = s.astype("int64").where(s.notna())
            if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                codes = s.astype("category").cat.codes.to_numpy()
                if (codes >= 0).any():
                    categorical.append(codes)
            else:
                v = s.to_numpy(dtype=float)
                valid = ~np.isnan(v)
                if valid.sum() < 2:
                    continue
                v_range = v[valid].max() - v[valid].min()
                if v_range < 1e-10:
                    continue
                numerical.append((v - v[valid].min()) / v_range)

        n = len(metadata)
        return MetadataFeatures(
            categorical=np.column_stack(categorical) if categorical else np.empty((n, 0), int),
            numerical=np.column_stack(numerical) if numerical else np.empty((n, 0)),
        )

    @staticmethod
    def _metadata_edge_similarity(
        metadata: "MetadataFeatures | csr_matrix",
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Metadata similarity in [0, 1] for the given node pairs."""
        if not isinstance(metadata, MetadataFeatures):
            # User-supplied sparse adjacency
            m = csr_matrix(metadata)
            sims = np.asarray(m[rows, cols]).ravel()
            max_val = m.max() if m.nnz > 0 else 0
            return sims / max_val if max_val > 0 else sims

        n_cols = metadata.categorical.shape[1] + metadata.numerical.shape[1]
        if n_cols == 0:
            return np.zeros(rows.size)
        sims = np.zeros(rows.size)
        for c in range(metadata.categorical.shape[1]):
            codes = metadata.categorical[:, c]
            sims += (codes[rows] == codes[cols]) & (codes[rows] >= 0)
        for c in range(metadata.numerical.shape[1]):
            v = metadata.numerical[:, c]
            sim = 1.0 - np.abs(v[rows] - v[cols])
            sims += np.where(sim > 0.8, sim, 0.0)  # NaN compares False -> 0
        return sims / n_cols

    def build_multiview_graph(
        self,
        semantic_embeddings: np.ndarray,
        lexical_matrix: csr_matrix | None = None,
        metadata_graph: "MetadataFeatures | csr_matrix | None" = None,
        weights: dict[str, float] | None = None,
        lexical_adj: csr_matrix | None = None,
    ) -> "igraph.Graph":
        """
        Build combined multi-view graph.

        Fuses semantic, lexical, and metadata views into a single graph
        for robust community detection.

        Parameters
        ----------
        semantic_embeddings : np.ndarray
            Document embeddings.
        lexical_matrix : csr_matrix, optional
            TF-IDF matrix for lexical view.
        metadata_graph : MetadataFeatures or csr_matrix, optional
            Output of ``build_metadata_graph`` (or a pre-computed metadata
            adjacency).  Reweights the semantic/lexical edges; it does not
            add new edges.
        weights : dict, optional
            Weights for each view. Keys: "semantic", "lexical", "metadata"
        lexical_adj : csr_matrix, optional
            Pre-computed lexical adjacency matrix. When provided, skips
            ``build_lexical_graph()`` -- useful for iterative refinement
            where the lexical graph never changes.

        Returns
        -------
        graph : igraph.Graph
            Combined weighted graph.
        """
        import igraph as ig
        from scipy.sparse import triu

        weights = weights or {"semantic": 0.5, "lexical": 0.3, "metadata": 0.2}
        n_samples = semantic_embeddings.shape[0]

        # Build semantic graph
        if self.graph_type == "knn":
            semantic_adj = self.build_knn_graph(semantic_embeddings)
        elif self.graph_type == "mutual_knn":
            semantic_adj = self.build_mutual_knn_graph(semantic_embeddings)
        elif self.graph_type == "snn":
            semantic_adj = self.build_snn_graph(semantic_embeddings)
        else:  # hybrid
            semantic_adj = self.build_hybrid_graph(semantic_embeddings)

        # Symmetrize (a plain kNN graph is directed) and normalize
        semantic_adj = semantic_adj.maximum(semantic_adj.T).tocsr()
        if semantic_adj.max() > 0:
            semantic_adj = semantic_adj / semantic_adj.max()

        # Determine which views are active and re-normalize weights
        active_weights = {"semantic": weights.get("semantic", 0.5)}
        has_lexical = lexical_matrix is not None and weights.get("lexical", 0) > 0
        has_metadata = metadata_graph is not None and weights.get("metadata", 0) > 0

        if has_lexical:
            active_weights["lexical"] = weights["lexical"]
        if has_metadata:
            active_weights["metadata"] = weights["metadata"]

        # Re-normalize so active weights sum to 1.0
        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            active_weights = {k: v / weight_sum for k, v in active_weights.items()}

        # Start with semantic
        combined_adj = active_weights["semantic"] * semantic_adj

        # Add lexical if available
        if has_lexical:
            if lexical_adj is None:
                lexical_adj = self.build_lexical_graph(lexical_matrix)
            if lexical_adj.max() > 0:
                lexical_adj = lexical_adj / lexical_adj.max()
            combined_adj = combined_adj + active_weights["lexical"] * lexical_adj

            # Consensus bonus: edges present in BOTH semantic and lexical
            # views are more reliable -- give them a small boost.
            # Use element-wise minimum (overlap strength) as the bonus.
            overlap = semantic_adj.minimum(lexical_adj)
            if overlap.nnz > 0:
                combined_adj = combined_adj + 0.1 * overlap

        # Undirected edge list (upper triangle; matrix is symmetric)
        upper = triu(combined_adj.tocsr(), k=1).tocoo()
        rows, cols, edge_weights = upper.row, upper.col, upper.data

        # Metadata: reweight existing edges
        if has_metadata:
            meta_sim = self._metadata_edge_similarity(metadata_graph, rows, cols)
            edge_weights = edge_weights + active_weights["metadata"] * meta_sim

        keep = edge_weights > 0
        graph = ig.Graph(
            n=n_samples,
            edges=np.column_stack([rows[keep], cols[keep]]).tolist(),
            directed=False,
        )
        graph.es["weight"] = edge_weights[keep].tolist()

        return graph

    def get_feature_names(self) -> list[str]:
        """Get TF-IDF feature names (for keyword extraction)."""
        if hasattr(self._tfidf_vectorizer, "get_feature_names_out"):
            return list(self._tfidf_vectorizer.get_feature_names_out())
        return list(self._tfidf_vectorizer.get_feature_names())
