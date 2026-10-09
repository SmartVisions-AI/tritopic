"""
Consensus Leiden Clustering
============================

Robust community detection with:
- Leiden algorithm (better than Louvain)
- Consensus clustering for stability
- Resolution parameter tuning
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score


class ConsensusLeiden:
    """
    Leiden clustering with consensus for stability.
    
    Runs multiple Leiden clusterings with different seeds and combines
    results using consensus clustering. This dramatically improves
    reproducibility and reduces sensitivity to random initialization.
    
    Parameters
    ----------
    resolution : float
        Resolution parameter for Leiden. Higher = more clusters. Default: 1.0
    n_runs : int
        Number of consensus runs. Default: 10
    random_state : int
        Random seed for reproducibility. Default: 42
    consensus_threshold : float
        Minimum agreement ratio for consensus. Default: 0.5
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_runs: int = 10,
        random_state: int = 42,
        consensus_threshold: float = 0.5,
    ):
        self.resolution = resolution
        self.n_runs = n_runs
        self.random_state = random_state
        self.consensus_threshold = consensus_threshold
        
        self.labels_: np.ndarray | None = None
        self.stability_score_: float | None = None
        self._all_partitions: list[np.ndarray] = []
    
    def fit_predict(
        self,
        graph: "igraph.Graph",
        min_cluster_size: int = 5,
        resolution: float | None = None,
    ) -> np.ndarray:
        """
        Fit Leiden clustering with consensus.

        Parameters
        ----------
        graph : igraph.Graph
            Input graph with edge weights.
        min_cluster_size : int
            Minimum cluster size. Smaller clusters become outliers.
        resolution : float, optional
            Override default resolution.

        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        res = self.resolution if resolution is None else resolution

        self._all_partitions = self._run_leiden(graph, res, self.n_runs)

        # Compute consensus
        self.labels_ = self._compute_consensus(graph, self._all_partitions, res)

        # Handle small clusters as outliers
        self.labels_ = self._handle_small_clusters(self.labels_, min_cluster_size)

        # Compute stability score
        self.stability_score_ = self._compute_stability()

        return self.labels_

    def _run_leiden(
        self,
        graph: "igraph.Graph",
        resolution: float,
        n_runs: int,
        weights: str = "weight",
    ) -> list[np.ndarray]:
        """Run Leiden *n_runs* times with consecutive seeds."""
        import leidenalg as la

        return [
            np.array(
                la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights=weights,
                    resolution_parameter=resolution,
                    seed=self.random_state + run,
                ).membership
            )
            for run in range(n_runs)
        ]

    def _compute_consensus(
        self,
        graph: "igraph.Graph",
        partitions: list[np.ndarray],
        resolution: float,
        max_rounds: int = 5,
    ) -> np.ndarray:
        """
        Consensus partition via edge-restricted co-occurrence
        (Lancichinetti & Fortunato, 2012).

        For every graph edge (i, j) the agreement is the fraction of runs
        that put i and j in the same cluster.  Edges with agreement below
        ``consensus_threshold`` are dropped (each node keeps its most
        consistent edge so nothing gets isolated), the remaining edges are
        reweighted by agreement, and Leiden is re-run on this consensus
        graph until all runs agree.

        Only O(n_edges) memory/time -- the previous dense n x n
        co-occurrence matrix plus average-linkage was O(n^2) memory and
        dominated runtime beyond ~10k documents.
        """
        import igraph as ig

        n_nodes = graph.vcount()
        edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        base_w = np.asarray(graph.es["weight"], dtype=float) if graph.ecount() else np.zeros(0)
        src, dst = edges[:, 0], edges[:, 1]

        prev_keep = None
        for _ in range(max_rounds):
            if all(np.array_equal(partitions[0], p) for p in partitions[1:]):
                break
            P = np.vstack(partitions)
            agree = (P[:, src] == P[:, dst]).mean(axis=0)

            # Every node keeps its highest-agreement edge
            node_best = np.zeros(n_nodes)
            np.maximum.at(node_best, src, agree)
            np.maximum.at(node_best, dst, agree)
            keep = (
                (agree >= self.consensus_threshold)
                | (agree >= node_best[src])
                | (agree >= node_best[dst])
            ) & (agree > 0)
            if prev_keep is not None and np.array_equal(keep, prev_keep):
                break  # consensus graph is stable; remaining disagreement is tie-level noise
            prev_keep = keep

            consensus = ig.Graph(n=n_nodes, edges=edges[keep].tolist(), directed=False)
            consensus.es["weight"] = (agree[keep] * base_w[keep]).tolist()
            partitions = self._run_leiden(consensus, resolution, len(partitions))

        # Pick the partition that agrees most with the others (medoid)
        if len(partitions) == 1:
            return partitions[0]
        scores = [
            np.mean([adjusted_rand_score(p, q) for q in partitions]) for p in partitions
        ]
        return partitions[int(np.argmax(scores))]

    @staticmethod
    def _fallback_best_partition(partitions: list[np.ndarray]) -> np.ndarray:
        """Pick the partition with the highest average ARI against all others."""
        best_labels = partitions[0]
        best_score = -1
        for p in partitions:
            avg = np.mean([adjusted_rand_score(p, q) for q in partitions])
            if avg > best_score:
                best_score = avg
                best_labels = p
        return best_labels
    
    def predict_single(
        self,
        graph: "igraph.Graph",
        min_cluster_size: int = 5,
        resolution: float | None = None,
    ) -> np.ndarray:
        """Single Leiden run without consensus — fast path for resolution tuning.

        Use this when the resolution has already been optimised by binary search
        and consensus clustering would produce a near-dense co-occurrence graph
        (few large clusters).

        Parameters
        ----------
        graph : igraph.Graph
            Input graph with edge weights.
        min_cluster_size : int
            Minimum cluster size. Smaller clusters become outliers.
        resolution : float, optional
            Override default resolution.

        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        import leidenalg as la

        res = self.resolution if resolution is None else resolution
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=self.random_state,
        )
        labels = np.array(partition.membership)
        return self._handle_small_clusters(labels, min_cluster_size)

    def _handle_small_clusters(
        self,
        labels: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """Mark small clusters as outliers (-1)."""
        result = labels.copy()
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            
            size = np.sum(labels == cluster_id)
            if size < min_size:
                result[labels == cluster_id] = -1
        
        # Relabel to consecutive integers
        unique_labels = sorted([l for l in np.unique(result) if l != -1])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        label_map[-1] = -1
        
        result = np.array([label_map[l] for l in result])
        
        return result
    
    def _compute_stability(self) -> float:
        """Compute stability score as average pairwise ARI."""
        if len(self._all_partitions) < 2:
            return 1.0
        
        ari_scores = []
        for i in range(len(self._all_partitions)):
            for j in range(i + 1, len(self._all_partitions)):
                ari = adjusted_rand_score(
                    self._all_partitions[i],
                    self._all_partitions[j]
                )
                ari_scores.append(ari)
        
        return float(np.mean(ari_scores))
    
    def find_optimal_resolution(
        self,
        graph: "igraph.Graph",
        resolution_range: tuple[float, float] = (0.1, 2.0),
        n_steps: int = 10,
        target_n_topics: int | None = None,
        min_cluster_size: int = 1,
    ) -> float:
        """
        Find optimal resolution parameter.

        When *target_n_topics* is given, uses binary search for much higher
        precision (O(log n) instead of O(n)).  Falls back to a linear sweep
        only when no target is specified.

        Parameters
        ----------
        graph : igraph.Graph
            Input graph.
        resolution_range : tuple
            Range of resolutions to search.
        n_steps : int
            Number of search steps (binary-search iterations when
            *target_n_topics* is given, linear sweep points otherwise).
        target_n_topics : int, optional
            If provided, find resolution closest to this number of topics.
        min_cluster_size : int
            Only clusters of at least this size count as topics (smaller
            ones become outliers in ``fit_predict``).

        Returns
        -------
        optimal_resolution : float
            Best resolution parameter.
        """
        import leidenalg as la

        def _n_clusters_at(res: float) -> int:
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=self.random_state,
            )
            sizes = np.bincount(partition.membership)
            return int(np.sum(sizes >= min_cluster_size))

        if target_n_topics is not None:
            # Binary search: higher resolution → more clusters
            # Bisect in log-space: the topic count scales roughly with
            # log(resolution), so a linear midpoint wastes most probes.
            lo, hi = resolution_range
            # Widen the bracket until it contains the target (a fixed range
            # silently missed targets on very clean or very noisy graphs).
            # Only widen while the count still moves towards the target: with
            # min_cluster_size the count drops again at very high resolutions
            # (clusters fragment below the minimum size).
            n_hi = _n_clusters_at(hi)
            for _ in range(12):
                if n_hi >= target_n_topics:
                    break
                n_next = _n_clusters_at(hi * 4)
                if n_next <= n_hi:
                    break
                lo, hi, n_hi = hi, hi * 4, n_next
            n_lo = _n_clusters_at(lo)
            for _ in range(12):
                if n_lo <= target_n_topics:
                    break
                n_next = _n_clusters_at(lo / 4)
                if n_next >= n_lo:
                    break
                lo, hi, n_lo = lo / 4, lo, n_next
            best_res, best_diff = lo, abs(_n_clusters_at(lo) - target_n_topics)

            for _ in range(n_steps):
                mid = float(np.sqrt(lo * hi))
                n_clust = _n_clusters_at(mid)
                diff = abs(n_clust - target_n_topics)

                if diff < best_diff:
                    best_diff = diff
                    best_res = mid

                if n_clust == target_n_topics:
                    return mid
                elif n_clust < target_n_topics:
                    lo = mid
                else:
                    hi = mid

            return best_res
        else:
            # Linear sweep for maximum modularity
            resolutions = np.linspace(resolution_range[0], resolution_range[1], n_steps)
            best_res = resolutions[0]
            best_mod = -float("inf")

            for res in resolutions:
                partition = la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=res,
                    seed=self.random_state,
                )
                if partition.modularity > best_mod:
                    best_mod = partition.modularity
                    best_res = res

            return best_res


class HDBSCANClusterer:
    """
    Alternative clustering using HDBSCAN.
    
    Useful for datasets with varying density or many outliers.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
    
    def fit_predict(
        self,
        embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit HDBSCAN clustering.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings (optionally reduced with UMAP first).
            
        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        import hdbscan
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            **kwargs,
        )
        
        self.labels_ = clusterer.fit_predict(embeddings)
        self.probabilities_ = clusterer.probabilities_
        
        return self.labels_
