# TriTopic v2.1 — Complete Technical Documentation

## Tri-Modal Graph-Based Topic Modeling with Iterative Refinement

**Version 2.1.0** | February 2026

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [The Full Pipeline — Step by Step](#2-the-full-pipeline--step-by-step)
   - 2.1 [Step 1: Document Embedding](#21-step-1-document-embedding)
   - 2.2 [Step 1.5: Dimensionality Reduction](#22-step-15-dimensionality-reduction)
   - 2.3 [Step 2: Lexical View (TF-IDF)](#23-step-2-lexical-view-tf-idf)
   - 2.4 [Step 3: Metadata View (optional)](#24-step-3-metadata-view-optional)
   - 2.5 [Step 4: Multi-View Graph Fusion](#25-step-4-multi-view-graph-fusion)
   - 2.6 [Step 5: Consensus Leiden Clustering](#26-step-5-consensus-leiden-clustering)
   - 2.7 [Step 6: Iterative Refinement](#27-step-6-iterative-refinement)
   - 2.8 [Step 7: Keyword Extraction & Topic Info](#28-step-7-keyword-extraction--topic-info)
   - 2.9 [Step 8: Centroid Computation & Probabilities](#29-step-8-centroid-computation--probabilities)
   - 2.10 [Step 9: Target Topic Count (Bidirectional Resolution Search)](#210-step-9-target-topic-count-bidirectional-resolution-search)
3. [Graph Construction — In Detail](#3-graph-construction--in-detail)
   - 3.1 [kNN Graph](#31-knn-graph)
   - 3.2 [Mutual kNN Graph](#32-mutual-knn-graph)
   - 3.3 [Shared Nearest Neighbors (SNN)](#33-shared-nearest-neighbors-snn)
   - 3.4 [Hybrid Graph (Default)](#34-hybrid-graph-default)
   - 3.5 [Lexical Graph](#35-lexical-graph)
   - 3.6 [Metadata Graph](#36-metadata-graph)
   - 3.7 [Multi-View Fusion with Consensus Bonus](#37-multi-view-fusion-with-consensus-bonus)
4. [Consensus Clustering — In Detail](#4-consensus-clustering--in-detail)
   - 4.1 [Multiple Leiden Runs](#41-multiple-leiden-runs)
   - 4.2 [Sparse Co-Occurrence Matrix](#42-sparse-co-occurrence-matrix)
   - 4.3 [Hierarchical Consensus Cut](#43-hierarchical-consensus-cut)
   - 4.4 [Small-Cluster Removal](#44-small-cluster-removal)
   - 4.5 [Stability Score](#45-stability-score)
5. [Iterative Refinement — In Detail](#5-iterative-refinement--in-detail)
   - 5.1 [Distance-Aware Centroid Pulling](#51-distance-aware-centroid-pulling)
   - 5.2 [Decaying Blend Factor](#52-decaying-blend-factor)
   - 5.3 [Convergence Detection](#53-convergence-detection)
6. [Bidirectional Resolution Search](#6-bidirectional-resolution-search)
7. [Post-Fit Operations](#7-post-fit-operations)
   - 7.1 [Outlier Reduction](#71-outlier-reduction)
   - 7.2 [Topic Merging (Size-Aware)](#72-topic-merging-size-aware)
   - 7.3 [Manual Topic Merging](#73-manual-topic-merging)
8. [Keyword Extraction Methods](#8-keyword-extraction-methods)
   - 8.1 [c-TF-IDF (Default)](#81-c-tf-idf-default)
   - 8.2 [BM25](#82-bm25)
   - 8.3 [KeyBERT](#83-keybert)
9. [Prediction on New Documents](#9-prediction-on-new-documents)
10. [Soft Topic Probabilities](#10-soft-topic-probabilities)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Complete Configuration Reference](#12-complete-configuration-reference)
13. [Key Design Decisions & Rationale](#13-key-design-decisions--rationale)
14. [Benchmark Results](#14-benchmark-results)

---

## 1. Design Philosophy

TriTopic is built on three convictions:

1. **No single view is sufficient.** Embeddings capture semantics but blur lexical details. TF-IDF captures term specificity but misses synonyms. Metadata captures structure but not content. TriTopic fuses all three into a unified graph where the strengths of one view compensate for the weaknesses of another.

2. **Stochastic algorithms need stabilization.** A single Leiden run depends on random initialization. Running it once and accepting the result is like flipping a coin and calling it science. TriTopic runs Leiden multiple times and builds a consensus, producing near-deterministic partitions (cross-seed NMI standard deviation: 0.007).

3. **Every document deserves a topic.** Density-based methods like HDBSCAN discard "difficult" documents as outliers (up to 29% of the corpus). TriTopic assigns every document to its most plausible topic via graph-based clustering, achieving 100% coverage without sacrificing quality.

---

## 2. The Full Pipeline — Step by Step

When `model.fit(documents)` is called, the following happens in order:

### 2.1 Step 1: Document Embedding

**What:** Each document is encoded into a dense vector using a Sentence-Transformer model.

**How:**
```
documents → SentenceTransformer("all-MiniLM-L6-v2") → embeddings (N × 384)
```

**Default model:** `all-MiniLM-L6-v2` (384 dimensions, fast, good quality).

**Details:**
- Encoding is batched (`batch_size=32`) to control memory usage.
- All embeddings are L2-normalized, so cosine similarity = dot product.
- The embedding engine is lazy-loaded: the model is not downloaded until the first `encode()` call.
- If pre-computed embeddings are passed via `fit(documents, embeddings=my_embeddings)`, this step is skipped entirely.

**Rationale:** Sentence-Transformers produce semantically meaningful vectors where "automobile" and "car" are close, even though they share no characters. This is the foundation of TriTopic's semantic understanding.

### 2.2 Step 1.5: Dimensionality Reduction

**What:** High-dimensional embeddings (384-768d) are projected to ~10 dimensions before graph construction.

**How:**
```
embeddings (N × 384) → UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine") → reduced_embeddings (N × 10)
```

**Details:**
- `min_dist=0.0` is deliberately set for clustering (not visualization).
- The fitted UMAP reducer is stored so new documents can be projected during `transform()`.
- Both the full and reduced embeddings are kept. Reduced are used only for graph construction; full are used for centroids, keywords, and probabilities.
- Alternative: PaCMAP can be used instead of UMAP (`dim_reduction_method="pacmap"`).

**Rationale:** kNN in 384 dimensions suffers from the curse of dimensionality: distances concentrate and all points appear equidistant. Reducing to 10d dramatically improves neighbor quality. The kNN graph built on 10d embeddings is cleaner and produces better clusters. This is not lossy compression — the full embeddings are preserved for all downstream operations.

### 2.3 Step 2: Lexical View (TF-IDF)

**What:** A sparse term-frequency matrix capturing exact word usage patterns.

**How:**
```
documents → TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),     # unigrams and bigrams
    min_df=2,               # ignore words appearing in < 2 docs
    max_df=0.95,            # ignore words appearing in > 95% of docs
    sublinear_tf=True       # log(1 + tf) instead of raw tf
) → lexical_matrix (N × V, sparse)
```

**Details:**
- `sublinear_tf=True` applies `log(1 + tf)` dampening, preventing long documents from dominating through sheer word count.
- The matrix is stored as `self.lexical_matrix_` and reused for keyword extraction.
- The vectorizer vocabulary is shared between graph construction and keyword extraction for consistency.

**Rationale:** Embeddings blur lexical distinctions. "Breakfast buffet" and "dinner service" both map to similar "food/dining" embeddings. TF-IDF preserves the distinction because the actual tokens differ. By building a separate lexical similarity graph and fusing it with the semantic graph, TriTopic gets both semantic depth and lexical precision.

### 2.4 Step 3: Metadata View (optional)

**What:** A similarity graph based on structured document attributes (source, category, date, etc.).

**How (if enabled and metadata provided):**

For each metadata column:
- **Categorical columns:** Documents sharing the same category value get an edge. Implemented as sparse indicator matrix multiplication: `M @ M.T` where M is the one-hot encoding.
- **Numerical columns:** Normalized to [0,1], then kNN with threshold: only pairs with similarity > 0.8 get an edge.

The per-column adjacency matrices are summed and normalized to [0,1].

**Rationale:** In many practical applications, documents with the same author, from the same time period, or tagged with the same category are more likely to share a topic. This view adds structural priors that pure content analysis misses.

### 2.5 Step 4: Multi-View Graph Fusion

**What:** The semantic, lexical, and metadata similarities are combined into a single igraph Graph.

**How:**

1. Build semantic graph from reduced embeddings using the configured graph type (default: hybrid MkNN + SNN).
2. Build lexical graph from TF-IDF matrix using mutual kNN.
3. Normalize both to [0, 1] by dividing by their respective maximum weight.
4. Determine active views (skip views that are disabled or have no data).
5. Re-normalize weights so active views sum to 1.0.
6. Combine: `combined = w_sem * semantic + w_lex * lexical + w_meta * metadata`
7. **Consensus bonus:** Add `+0.1` to edges that appear in *both* semantic and lexical graphs. This rewards structurally agreed-upon connections.
8. Convert to igraph.Graph with edge weights.

**Default weights:** semantic=0.5, lexical=0.3, metadata=0.2. If metadata is not used, weights re-normalize to semantic=0.625, lexical=0.375.

**Rationale:** Each view captures different aspects of document similarity. The weighted fusion ensures that two documents are strongly connected only if multiple views agree. The consensus bonus further rewards edges supported by both embedding-based and term-based evidence.

See [Section 3](#3-graph-construction--in-detail) for detailed graph construction algorithms.

### 2.6 Step 5: Consensus Leiden Clustering

**What:** The fused graph is partitioned into communities using the Leiden algorithm, run multiple times for stability.

**How:**

1. Run Leiden `n_consensus_runs` times (default: 10) with different random seeds.
2. Build a co-occurrence matrix: `C[i,j]` = fraction of runs where documents i and j were in the same cluster.
3. Apply hierarchical clustering (average linkage) on `1 - C` to produce a consensus partition.
4. Select the cut that maximizes the average ARI with all individual Leiden runs.
5. Mark clusters smaller than `min_cluster_size` (default: 5) as outliers (-1).

**Rationale:** A single Leiden run is unstable. Two runs with different seeds can produce substantially different partitions. The consensus approach filters out random noise: if two documents are consistently co-clustered across 10 runs, their grouping is structural, not accidental.

See [Section 4](#4-consensus-clustering--in-detail) for the full algorithm.

### 2.7 Step 6: Iterative Refinement

**What:** Embeddings are softly blended toward their topic centroid, then the graph and clustering are re-run. This tightens cluster boundaries.

**How (if `use_iterative_refinement=True`, default):**

```
For iteration = 1 to max_iterations (default: 5):
    1. Compute topic centroids from current embeddings
    2. For each document:
         cos_sim = cosine_similarity(document, its_centroid)
         per_doc_blend = blend_factor * sqrt(cos_sim)
         refined = (1 - per_doc_blend) * original + per_doc_blend * centroid
    3. L2-normalize refined embeddings
    4. Re-reduce dimensions (UMAP transform)
    5. Rebuild graph and re-cluster
    6. Check convergence: ARI(current, previous) > 0.95 → stop
```

**Blend factor decay:** Starts at 0.3 (aggressive), decreases to ~0.1 (conservative) across iterations:
```
blend = 0.3 - 0.2 * (iteration / (max_iterations - 1))
```

**Rationale:** Initial clustering creates reasonable but imperfect boundaries. By pulling documents toward their centroids and re-clustering, the boundaries sharpen. The distance-aware blend is crucial: core documents (high cosine similarity to centroid) receive the full pull, while borderline documents receive a weaker pull to avoid forcing ambiguous documents into the wrong cluster.

See [Section 5](#5-iterative-refinement--in-detail) for the full algorithm.

### 2.8 Step 7: Keyword Extraction & Topic Info

**What:** For each discovered topic, extract representative keywords and find representative documents.

**How:**

1. For each non-outlier topic:
   - Collect all documents in the topic.
   - Extract keywords using the configured method (default: c-TF-IDF).
   - Find representative documents: documents with highest cosine similarity to the topic centroid.
   - Create a `TopicInfo` object storing: topic_id, size, keywords, keyword_scores, representative_docs.

2. Topics are sorted by size (descending), with the outlier topic (-1) listed last.

**Rationale:** Keywords define the human-readable interface of each topic. Representative documents provide concrete examples. Both are essential for interpretability.

See [Section 8](#8-keyword-extraction-methods) for keyword extraction details.

### 2.9 Step 8: Centroid Computation & Probabilities

**What:** Compute topic centroid embeddings and soft probability distributions.

**Centroids:**
```
For each non-outlier topic:
    centroid = mean(original_embeddings[topic_members])
```

**Key detail:** Centroids are computed from **original (unrefined) embeddings**, not the iteratively refined ones. This ensures that `transform()` on new documents operates in the same embedding space.

**Probabilities:**
```
similarities = cosine_similarity(original_embeddings, topic_centroids)
probabilities = softmax(similarities * temperature, axis=1)
```

Default temperature: 5.0 (fairly sharp peaks, confident assignments).

**Rationale:** Hard labels (argmax) lose information about borderline documents. Soft probabilities reveal documents that belong partially to multiple topics. The temperature parameter controls how "peaked" the distribution is: higher temperature = sharper peaks = more confident assignments.

### 2.10 Step 9: Target Topic Count (Bidirectional Resolution Search)

**What:** If the user specifies `n_topics=K`, TriTopic searches for the Leiden resolution that naturally produces K clusters.

**How:**

1. Compare current topic count to target.
2. **If target > current:** Search higher resolutions in range `[resolution, resolution * 10]`.
3. **If target < current:** Search lower resolutions in range `[0.001, resolution]`.
4. Use binary search (20 steps) to find the resolution closest to the target.
5. Re-cluster at the best resolution found.
6. If the result still overshoots: fall back to `reduce_topics()` for final adjustment.

**Rationale:** The naive approach — cluster with default resolution, then merge topics down — produces poor results because greedy merging destroys natural cluster structure. For example, BBC News has 5 natural categories. Clustering at high resolution produces 12 topics, and merging down to 5 gives NMI ~0.04. Clustering directly at a lower resolution produces 5 natural clusters with NMI ~0.70. The bidirectional search finds the resolution that produces the right number of clusters organically.

See [Section 6](#6-bidirectional-resolution-search) for the full algorithm.

---

## 3. Graph Construction — In Detail

### 3.1 kNN Graph

The simplest graph type. Each document connects to its k nearest neighbors.

```
For each document i:
    Find k nearest neighbors by cosine distance
    Add directed edge i → j with weight = 1 - distance(i, j)
```

**Problem:** Asymmetric. Document A might consider B a neighbor, but B might not consider A a neighbor. These one-way connections can bridge unrelated clusters.

### 3.2 Mutual kNN Graph

Only keeps edges where both endpoints consider each other neighbors.

```
(i, j) is an edge  ⟺  j ∈ kNN(i)  AND  i ∈ kNN(j)
```

**Implementation (vectorized):**
1. Build directed kNN adjacency as sparse matrix.
2. Compute element-wise minimum with transpose: `mutual = knn.minimum(knn.T)`.
3. Average forward and reverse weights: `weight = (knn[i,j] + knn[j,i]) / 2`.

**Effect:** Removes 40-60% of edges compared to kNN. The surviving edges represent strong, bidirectional similarity. This eliminates noise bridges between unrelated clusters.

### 3.3 Shared Nearest Neighbors (SNN)

Edge weight = number of shared neighbors between two documents, normalized by k.

```
w(i, j) = |kNN(i) ∩ kNN(j)| / k
```

**Implementation:**
1. For each document, store its kNN set.
2. For each pair of mutual neighbors, compute set intersection size.
3. Normalize by k.

**Insight:** Two documents in the same dense region share many neighbors. Two documents in different regions share few. SNN captures topological density, not raw distance.

### 3.4 Hybrid Graph (Default)

Weighted combination of mutual kNN and SNN.

```
hybrid = (1 - snn_weight) * mutual_knn + snn_weight * snn
```

Default `snn_weight=0.5` gives equal contribution from both.

**Optimization:** The kNN computation is shared — computed once and reused by both mutual kNN and SNN construction. This avoids redundant work.

**Rationale:** Mutual kNN provides direct similarity signals (high-weight = similar content). SNN provides structural signals (high-weight = same neighborhood). The combination captures both aspects.

### 3.5 Lexical Graph

Built from TF-IDF vectors using mutual kNN with cosine distance.

```
tfidf_matrix (N × V) → NearestNeighbors(metric="cosine") → mutual kNN graph
```

Documents that share the same rare, topic-specific terms get strong connections. Documents that only share common words get weak connections (filtered out by mutual kNN).

### 3.6 Metadata Graph

Built from structured attributes.

**Categorical columns:** `M @ M.T` where M is the one-hot encoding. Creates edges between all documents sharing a category value.

**Numerical columns:** kNN with similarity threshold > 0.8. Only documents with very similar numerical values get connected.

### 3.7 Multi-View Fusion with Consensus Bonus

The three graphs are combined with configurable weights:

```
combined = w_sem * semantic_adj + w_lex * lexical_adj + w_meta * metadata_adj
```

**Consensus Bonus:** Edges that exist in both the semantic and lexical graph receive an additional `+0.1` weight boost:

```
consensus_edges = semantic_adj.multiply(lexical_adj)  # element-wise: non-zero where both exist
combined += 0.1 * (consensus_edges > 0)
```

**Rationale:** If both the embedding similarity and the word-overlap similarity agree that two documents are related, that connection is very likely real. The bonus amplifies these high-confidence edges, improving cluster purity.

**Deduplication:** The combined sparse matrix may have duplicate edges (from overlapping views). These are resolved by keeping the maximum weight for each (i, j) pair.

---

## 4. Consensus Clustering — In Detail

### 4.1 Multiple Leiden Runs

```python
for run in range(n_consensus_runs):  # default: 10
    seed = random_state + run
    partition = leidenalg.find_partition(
        graph,
        la.RBConfigurationVertexPartition,  # resolution-based modularity
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(partition.membership)
    all_partitions.append(labels)
```

Each run uses a different seed, producing a different (but related) partition. The RBConfiguration variant uses the resolution parameter to control granularity.

### 4.2 Sparse Co-Occurrence Matrix

For each partition, we build a sparse indicator matrix M (N × K) where M[i, c] = 1 if document i is in cluster c.

```
co_occur = sum over runs of (M @ M.T)
```

`M @ M.T` is a sparse N×N matrix where entry (i, j) = 1 if documents i and j are in the same cluster in that run. Summing across all runs and dividing by n_runs gives the co-occurrence probability.

**Rationale for sparse computation:** Directly building a dense N×N matrix is O(N²) in memory. The sparse indicator approach leverages the fact that cluster memberships are sparse (each document is in exactly one cluster), making the computation efficient for large corpora.

### 4.3 Hierarchical Consensus Cut

```
distance = 1.0 - co_occur_dense
distance = clip((distance + distance.T) / 2, 0, 1)  # ensure symmetry
condensed = squareform(distance)
Z = linkage(condensed, method="average")
```

The hierarchical tree is cut at the level that best matches the original partitions:

```
median_k = median(n_clusters across runs)
for n_clusters in range(median_k - 2, median_k + 3):
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    score = mean([ARI(labels, p) for p in all_partitions])
    keep best
```

**Fallback:** If hierarchical consensus fails, the partition with the highest average ARI to all other partitions is selected directly.

### 4.4 Small-Cluster Removal

Clusters with fewer than `min_cluster_size` (default: 5) documents are dissolved. Their members are marked as outliers (-1). Remaining clusters are relabeled to consecutive integers starting at 0.

**Rationale:** Very small clusters are typically noise or edge cases. Marking them as outliers (rather than forcing them into a topic) preserves the integrity of the remaining clusters. They can be reassigned later via `reduce_outliers()`.

### 4.5 Stability Score

```
stability = mean([ARI(partition_i, partition_j) for all pairs i < j])
```

This quantifies how consistent the Leiden runs are. Typical values:
- 0.90+ : Very stable (strong cluster structure)
- 0.70-0.90 : Moderately stable
- < 0.70 : Unstable (consider adjusting resolution or data preprocessing)

---

## 5. Iterative Refinement — In Detail

### 5.1 Distance-Aware Centroid Pulling

For each document in each topic:

```python
centroid = mean(embeddings[topic_members])
cos_sim = dot(normalized_embedding, normalized_centroid)
per_doc_blend = blend_factor * sqrt(max(0, cos_sim))
refined = (1 - per_doc_blend) * original + per_doc_blend * centroid
```

The `sqrt(cos_sim)` scaling is the key innovation:
- **Core documents** (cos_sim ≈ 0.9): blend ≈ 0.95 × blend_factor → strong pull toward centroid.
- **Borderline documents** (cos_sim ≈ 0.4): blend ≈ 0.63 × blend_factor → moderate pull.
- **Misplaced documents** (cos_sim ≈ 0.1): blend ≈ 0.32 × blend_factor → weak pull, protecting against reinforcing errors.

After blending, all embeddings are L2-normalized to maintain unit length.

### 5.2 Decaying Blend Factor

```
blend = 0.3 - 0.2 * (iteration / (max_iterations - 1))
```

| Iteration | blend_factor | Effect |
|-----------|-------------|--------|
| 0 | 0.30 | Aggressive repositioning — move documents decisively |
| 1 | 0.25 | Still substantial movement |
| 2 | 0.20 | Moderate adjustment |
| 3 | 0.15 | Fine-tuning |
| 4 | 0.10 | Minimal adjustment — only subtle corrections |

**Rationale:** Early iterations need aggressive repositioning to fix initial misassignments. Later iterations should only make subtle corrections. Without decay, the algorithm can oscillate between partitions.

### 5.3 Convergence Detection

After each iteration, compute ARI between current and previous partition:

```python
ari = adjusted_rand_score(current_labels, previous_labels)
if ari >= convergence_threshold:  # default: 0.95
    break  # converged
```

Typical convergence: iteration 2-3 (ARI > 0.95 by the third pass).

**What the iteration history stores:**

```python
_iteration_history = [
    {"iteration": 1, "ari": 0.82, "n_topics": 14},
    {"iteration": 2, "ari": 0.94, "n_topics": 13},
    {"iteration": 3, "ari": 0.97, "n_topics": 13},  # converged
]
```

---

## 6. Bidirectional Resolution Search

When `n_topics` is set to an integer (e.g., `n_topics=5`):

```python
if target > current_n_topics:
    # Need MORE topics → search HIGHER resolutions
    search_range = (current_resolution, current_resolution * 10)
else:
    # Need FEWER topics → search LOWER resolutions
    search_range = (0.001, current_resolution)
```

The search uses `ConsensusLeiden.find_optimal_resolution()` with binary search (20 steps):

```
lo, hi = search_range
for step in range(20):
    mid = (lo + hi) / 2
    n_clusters = leiden_at_resolution(mid)
    if n_clusters < target: lo = mid
    else: hi = mid
    track best (closest to target)
```

If the binary search overshoots (e.g., target 5 but minimum achievable is 7), the model falls back to `reduce_topics()` to merge down the remaining difference.

**Why this matters:** On BBC News with 5 classes, clustering at the default resolution produces ~12 topics. The old approach merged 12 → 5 via greedy centroid-similarity merging, which produced NMI ≈ 0.04. The new approach finds a resolution that naturally produces 5 clusters, achieving NMI ≈ 0.70. This is a 17× improvement in cluster quality.

---

## 7. Post-Fit Operations

### 7.1 Outlier Reduction

**Strategy: "embeddings" (default)**
```python
for each outlier document:
    similarities = cosine_similarity(doc_embedding, all_topic_centroids)
    best_topic = argmax(similarities)
    if max_similarity > threshold:  # default: 0.35
        assign to best_topic
    else:
        keep as outlier
```

**Strategy: "neighbors"**
```python
for each outlier document:
    find k nearest non-outlier neighbors
    assign by majority vote of their labels
```

After reassignment, keywords, centroids, and probabilities are recomputed.

### 7.2 Topic Merging (Size-Aware)

When reducing to a target topic count via `reduce_topics(n)`:

```
while n_topics > target:
    1. Compute cosine similarity between all topic centroid pairs
    2. Apply size-aware penalty:
         sim[i,j] *= (min(size_i, size_j) / max(size_i, size_j))^0.3
    3. Find the pair with highest adjusted similarity
    4. Merge smaller topic into larger (keep larger's ID)
    5. Relabel all member documents
```

**Why size-aware?** Without the penalty, two large topics of 500 documents each might merge before two small topics of 20 documents each, simply because large topics tend to have similar centroids (regression to the mean). The `(min/max)^0.3` penalty prefers merging small topics into large ones, preserving major theme boundaries.

### 7.3 Manual Topic Merging

`merge_topics([2, 7])` explicitly merges topics 2 and 7. The larger topic's ID is kept. Keywords, centroids, and probabilities are recomputed.

---

## 8. Keyword Extraction Methods

### 8.1 c-TF-IDF (Default)

Class-based TF-IDF. All documents in a topic are concatenated into a single "class document."

```
1. Fit vocabulary on entire corpus (once, cached)
2. Compute IDF: log(N / (1 + doc_freq_per_term))
3. For each topic:
   - Concatenate all topic documents
   - Compute term frequencies
   - Normalize: tf / sum(tf)
   - Score: c-tfidf = normalized_tf × idf
4. Return top n_keywords by score
```

**Rationale:** c-TF-IDF identifies words that are frequent within a topic but rare in the overall corpus. A word like "basketball" might appear in only 3% of all documents (high IDF) but in 80% of a sports topic's documents (high TF), making it a strong topic keyword.

### 8.2 BM25

Okapi BM25 relevance scoring with specificity adjustment.

```
1. Tokenize all documents (custom tokenizer: lowercase, alpha-only, stopword filter)
2. Build BM25 index from corpus
3. For each word in topic vocabulary:
   - Compute average BM25 score within topic
   - Compute average BM25 score across corpus
   - Specificity = (topic_avg / corpus_avg) × log(1 + frequency)
4. Return top n by specificity
```

**Rationale:** BM25 is more robust to document length variation than TF-IDF. It also handles term saturation better — a word appearing 10 times vs. 100 times in a long document doesn't get 10× the weight.

### 8.3 KeyBERT

Embedding-based keyword extraction using Maximal Marginal Relevance.

```
1. Concatenate all topic documents
2. Extract candidate n-grams
3. Embed candidates and topic text
4. Score by cosine similarity to topic embedding
5. Apply MMR to maximize diversity (diversity=0.5)
6. Return top n
```

**Rationale:** KeyBERT captures semantic keywords that may not appear literally in the text. It also ensures keyword diversity through MMR, preventing near-duplicate keywords like "machine learning" and "learning machine" from both appearing.

---

## 9. Prediction on New Documents

```python
new_labels = model.transform(new_documents)
```

**Algorithm:**
1. Encode new documents using the same embedding model (full-dimensional, not reduced).
2. Compute cosine similarity between new embeddings and stored `topic_embeddings_` (centroids from original training embeddings).
3. Assign each document to the topic with highest similarity.
4. Mark as outlier (-1) if max similarity < `outlier_threshold` (default: 0.35).

**Key design choice:** Centroids are computed from **original (unrefined) embeddings**, not the iteratively refined ones. This ensures that new documents (which have not been refined) are compared in the same embedding space.

---

## 10. Soft Topic Probabilities

```python
proba = model.transform_proba(new_documents)  # shape: (n_docs, n_topics)
```

**Algorithm:**
```
similarities = cosine_similarity(embeddings, topic_centroids)  # (n_docs, n_topics)
probabilities = softmax(similarities × temperature, axis=1)    # temperature default: 5.0
```

**Temperature effect:**
- temperature=1.0: Relatively flat distributions. A document with similarities [0.8, 0.7, 0.6] gets probabilities [0.37, 0.34, 0.29].
- temperature=5.0 (default): Sharper peaks. Same similarities → [0.67, 0.24, 0.09].
- temperature=10.0: Very peaked. Same similarities → [0.88, 0.10, 0.02].

---

## 11. Evaluation Metrics

```python
metrics = model.evaluate()
```

Returns:
- `coherence_mean`: Mean NPMI coherence across topics. Range [-1, 1]. Higher = better.
- `coherence_std`: Standard deviation of per-topic coherence.
- `diversity`: Proportion of unique keywords across all topics. Range [0, 1]. Higher = topics are more distinct.
- `stability`: Average pairwise ARI across consensus Leiden runs. Range [-1, 1]. Higher = more reproducible.
- `n_topics`: Number of non-outlier topics found.
- `outlier_ratio`: Fraction of documents marked as outliers. 0.0 = perfect coverage.

---

## 12. Complete Configuration Reference

| Parameter | Default | Description | Rationale |
|-----------|---------|-------------|-----------|
| **Embedding** | | | |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-Transformer model | Best speed/quality tradeoff |
| `embedding_batch_size` | `32` | Encoding batch size | Memory control |
| **Dimensionality Reduction** | | | |
| `use_dim_reduction` | `True` | Reduce before graph building | Dramatically improves kNN quality |
| `reduced_dims` | `10` | Target dimensions | 10 retains most structure; <5 loses too much |
| `dim_reduction_method` | `"umap"` | UMAP or PaCMAP | UMAP is better studied for clustering |
| `umap_n_neighbors` | `15` | UMAP neighborhood size | Matches graph n_neighbors for consistency |
| `umap_min_dist` | `0.0` | UMAP minimum distance | 0.0 is optimal for clustering (not visualization) |
| **Graph Construction** | | | |
| `n_neighbors` | `15` | k for kNN graphs | 15 is standard; increase for larger corpora |
| `metric` | `"cosine"` | Distance metric | Cosine is standard for normalized embeddings |
| `graph_type` | `"hybrid"` | Graph algorithm | Hybrid combines MkNN stability + SNN robustness |
| `snn_weight` | `0.5` | SNN weight in hybrid | Equal contribution from both views |
| **Multi-View Fusion** | | | |
| `use_lexical_view` | `True` | Include TF-IDF view | Counters embedding blur |
| `use_metadata_view` | `False` | Include metadata view | Enable when metadata is available |
| `semantic_weight` | `0.5` | Semantic graph weight | Dominant view (contextual understanding) |
| `lexical_weight` | `0.3` | Lexical graph weight | Secondary view (term precision) |
| `metadata_weight` | `0.2` | Metadata graph weight | Supplementary view (structural priors) |
| **Clustering** | | | |
| `resolution` | `1.0` | Leiden resolution | Higher = more topics. 1.0 is balanced. |
| `n_consensus_runs` | `10` | Number of Leiden runs | 7-10 is sufficient for stability |
| `min_cluster_size` | `5` | Minimum topic size | Smaller → outlier. 5 prevents noise clusters. |
| **Iterative Refinement** | | | |
| `use_iterative_refinement` | `True` | Enable refinement loop | Significantly improves NMI (+5-15%) |
| `max_iterations` | `5` | Maximum iterations | Convergence usually by iteration 3 |
| `convergence_threshold` | `0.95` | ARI stopping criterion | 0.95 = partitions are practically identical |
| **Keywords** | | | |
| `n_keywords` | `10` | Keywords per topic | 10 gives good interpretability |
| `n_representative_docs` | `5` | Representative docs per topic | 5 shows topic breadth |
| `keyword_method` | `"ctfidf"` | Extraction method | c-TF-IDF is fast and effective |
| **Outlier Handling** | | | |
| `outlier_threshold` | `0.35` | Min similarity for topic assignment | Lower → more assignments, potentially noisier |
| **Probability** | | | |
| `softmax_temperature` | `5.0` | Softmax sharpness | 5.0 gives confident but not extreme probabilities |
| **Misc** | | | |
| `random_state` | `42` | Random seed | Reproducibility baseline |
| `verbose` | `True` | Print progress | |

---

## 13. Key Design Decisions & Rationale

### Why Leiden over HDBSCAN?

HDBSCAN produces outliers by design. In BERTopic, 10-30% of documents are discarded as noise. Leiden, operating on a graph, assigns every connected node to a community. By combining Leiden with consensus, TriTopic achieves both complete coverage and stable partitions.

### Why consensus clustering instead of a single run?

A single Leiden run depends on random initialization and can produce substantially different results with different seeds. Running 10 times and computing the co-occurrence matrix reveals which groupings are structural (consistent across runs) and which are artifacts (varying randomly). This is the difference between a measurement and a guess.

### Why not concatenate the views?

Concatenating a 384-dim semantic vector with a 10,000-dim TF-IDF vector would create a 10,384-dim space dominated by the sparse TF-IDF dimensions. Graph-based fusion avoids this: each view contributes a separate similarity signal, and the weights control their relative importance.

### Why refine embeddings iteratively?

Initial clusters have fuzzy boundaries. A document at the border between "AI Research" and "Software Engineering" might be assigned to either. By gently pulling it toward its assigned cluster's centroid and re-clustering, the model discovers whether this pull stabilizes (correct assignment) or the document switches clusters (incorrect assignment that gets corrected).

### Why use original embeddings for centroids and transform?

During iterative refinement, embeddings are modified to tighten clusters. These modifications are specific to the training data's cluster structure. New documents haven't undergone this refinement. If centroids were computed from refined embeddings, new documents would be compared in a different space. Using original embeddings for centroids ensures consistency.

### Why distance-aware blending?

Uniform blending (same pull for all documents) would drag borderline documents toward the wrong centroid if they were initially misassigned. The `sqrt(cosine_similarity)` scaling ensures that only confidently-assigned documents get fully pulled, while uncertain documents are treated conservatively.

### Why the consensus bonus for cross-view edges?

An edge supported by both semantic and lexical evidence is more reliable than one supported by only one view. The embedding might place two hotel reviews together because they're both positive, but TF-IDF would separate them because one mentions "breakfast" and the other "dinner." If both views agree that two documents are similar, that connection is very likely genuine.

### Why size-aware topic merging?

Without size-awareness, greedy merging by centroid similarity tends to merge large, broad topics first (their centroids are similar due to regression to the mean), destroying major thematic boundaries. The penalty `(min_size/max_size)^0.3` ensures small, peripheral topics are merged first, preserving the core structure.

### Why bidirectional resolution search?

The resolution parameter directly controls Leiden's granularity. Higher resolution produces more clusters. When the user requests fewer topics than the natural count, decreasing resolution is far superior to merging. Merging post-hoc destroys coherent cluster structure; lower resolution produces naturally coherent coarser partitions.

---

## 14. Benchmark Results

Evaluated on 4 standard datasets, 3 seeds per configuration, 5 topic counts per dataset (60 runs per model).

### Overall

| Model | NMI | Coherence | Coverage | Runtime |
|-------|-----|-----------|----------|---------|
| **TriTopic** | **0.575** | **0.341** | **1.000** | 62.6s |
| BERTopic | 0.513 | 0.233 | 0.808 | 10.9s |
| NMF | 0.416 | 0.330 | 1.000 | 3.6s |
| LDA | 0.299 | 0.161 | 1.000 | 8.6s |

### Per-Dataset NMI

| Dataset | TriTopic | BERTopic | NMF | LDA |
|---------|----------|----------|-----|-----|
| 20 Newsgroups | **0.532** | 0.519 | 0.319 | 0.158 |
| BBC News | **0.702** | 0.642 | 0.648 | 0.505 |
| AG News | **0.527** | 0.380 | 0.191 | 0.027 |
| Arxiv | **0.540** | 0.511 | 0.505 | 0.508 |

### Cross-Seed Stability

| Model | Mean NMI Std |
|-------|-------------|
| NMF | 0.005 |
| **TriTopic** | **0.007** |
| BERTopic | 0.011 |
| LDA | 0.021 |

TriTopic achieves the highest NMI on every dataset with 100% coverage and near-deterministic reproducibility.
