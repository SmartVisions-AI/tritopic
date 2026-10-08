"""
Evaluation Metrics for Topic Models
====================================

Provides standard metrics for evaluating topic model quality:
- Coherence (NPMI, CV)
- Diversity
- Stability (ARI between runs)
"""

from __future__ import annotations

import numpy as np
from itertools import combinations


def compute_coherence(
    keywords: list[str],
    documents: list[str],
    method: str = "npmi",
    window_size: int = 10,
) -> float:
    """
    Compute topic coherence based on document-level keyword co-occurrence.

    Parameters
    ----------
    keywords : list[str]
        Topic keywords (unigrams or n-grams).
    documents : list[str]
        Reference corpus for co-occurrence statistics -- normally the
        *whole* corpus, not only the topic's documents.
    method : str
        Coherence method: "npmi" (default), "uci", "umass"
    window_size : int
        Unused; co-occurrence is counted per document.  Kept for API
        compatibility.

    Returns
    -------
    coherence : float
        Coherence score (higher is better).
    """
    return compute_coherence_batch([keywords], documents, method=method)[0]


def compute_coherence_batch(
    topic_keywords: list[list[str]],
    documents: list[str],
    method: str = "npmi",
    language: str = "english",
) -> list[float]:
    """
    Coherence for many topics with a single pass over the reference corpus.

    Keywords are matched with the same analyzer used for keyword extraction
    (lowercase, stop words of *language* removed, n-grams), so bigram keywords
    such as "space shuttle" are counted instead of silently skipped.

    NPMI follows Bouma (2009): pairs that never co-occur score -1.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from tritopic.utils.stopwords import TOKEN_PATTERN, get_stopwords

    vocab = sorted({kw.lower() for kws in topic_keywords for kw in kws})
    if not vocab or not documents:
        return [0.0] * len(topic_keywords)

    max_n = max(len(term.split()) for term in vocab)
    vectorizer = CountVectorizer(
        vocabulary=vocab, ngram_range=(1, max_n), stop_words=get_stopwords(language), binary=True,
        token_pattern=TOKEN_PATTERN,
    )
    X = vectorizer.transform(documents)
    index = {term: i for i, term in enumerate(vocab)}
    return coherence_from_doc_term(
        X, [[index[kw.lower()] for kw in kws] for kws in topic_keywords], method=method
    )


def coherence_from_doc_term(
    doc_term,
    topic_term_ids: list[list[int]],
    method: str = "npmi",
) -> list[float]:
    """
    Coherence from an existing document-term matrix (counts or binary).

    *topic_term_ids* holds, per topic, the column indices of its keywords.
    Used by the model to score many candidate topic sets without
    re-tokenizing the corpus.
    """
    from scipy.sparse import csc_matrix

    used = sorted({i for ids in topic_term_ids for i in ids})
    if not used:
        return [0.0] * len(topic_term_ids)
    col = {c: j for j, c in enumerate(used)}
    X = csc_matrix(doc_term)[:, used]
    X.data = np.ones_like(X.data, dtype=np.float64)
    n_docs = X.shape[0]
    doc_freq = np.asarray(X.sum(axis=0)).ravel()
    co_freq = (X.T @ X).toarray()

    results = []
    for term_ids in topic_term_ids:
        ids = [col[i] for i in term_ids]
        scores = []
        for a, b in combinations(ids, 2):
            if a == b or doc_freq[a] == 0 or doc_freq[b] == 0:
                continue
            f_a, f_b, f_ab = doc_freq[a], doc_freq[b], co_freq[a, b]
            if method == "npmi":
                if f_ab == 0:
                    scores.append(-1.0)
                    continue
                p_ab = f_ab / n_docs
                pmi = np.log(p_ab / ((f_a / n_docs) * (f_b / n_docs)))
                scores.append(1.0 if p_ab >= 1.0 else pmi / -np.log(p_ab))
            elif method == "uci":
                p_ab = (f_ab + 1) / (n_docs + 1)
                scores.append(np.log(p_ab / (((f_a + 1) / (n_docs + 1)) * ((f_b + 1) / (n_docs + 1)))))
            elif method == "umass":
                scores.append(np.log((f_ab + 1) / f_b))
            else:
                raise ValueError(f"Unknown coherence method: {method!r}")
        results.append(float(np.mean(scores)) if scores else 0.0)
    return results


def compute_diversity(
    all_keywords: list[str],
    n_topics: int,
) -> float:
    """
    Compute topic diversity (proportion of unique keywords).
    
    Diversity measures how different topics are from each other.
    A model where every topic has the same keywords has diversity 0.
    
    Parameters
    ----------
    all_keywords : list[str]
        All keywords from all topics (flattened).
    n_topics : int
        Number of topics.
        
    Returns
    -------
    diversity : float
        Diversity score between 0 and 1 (higher is better).
    """
    if not all_keywords or n_topics == 0:
        return 0.0
    
    unique_keywords = set(kw.lower() for kw in all_keywords)
    
    # Diversity = unique keywords / total keywords
    diversity = len(unique_keywords) / len(all_keywords)
    
    return float(diversity)


def compute_stability(
    partitions: list[np.ndarray],
) -> float:
    """
    Compute clustering stability as average pairwise ARI.
    
    Parameters
    ----------
    partitions : list[np.ndarray]
        Multiple cluster assignments from different runs.
        
    Returns
    -------
    stability : float
        Average Adjusted Rand Index between partitions.
    """
    from sklearn.metrics import adjusted_rand_score
    
    if len(partitions) < 2:
        return 1.0
    
    ari_scores = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            ari = adjusted_rand_score(partitions[i], partitions[j])
            ari_scores.append(ari)
    
    return float(np.mean(ari_scores))


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for cluster quality.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings.
    labels : np.ndarray
        Cluster assignments.
        
    Returns
    -------
    silhouette : float
        Silhouette score between -1 and 1 (higher is better).
    """
    from sklearn.metrics import silhouette_score
    
    # Filter out outliers
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    
    unique_labels = np.unique(labels[mask])
    if len(unique_labels) < 2:
        return 0.0
    
    return float(silhouette_score(embeddings[mask], labels[mask]))


def compute_downstream_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    task: str = "classification",
) -> float:
    """
    Evaluate topic model by downstream task performance.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings.
    labels : np.ndarray
        Topic assignments.
    y_true : np.ndarray
        True labels for downstream task.
    task : str
        Task type: "classification" or "clustering"
        
    Returns
    -------
    score : float
        Task-specific score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, adjusted_rand_score
    from sklearn.model_selection import cross_val_score
    
    # Create topic features (one-hot + embedding)
    unique_topics = np.unique(labels[labels != -1])
    n_topics = len(unique_topics)

    # Create mapping from label to contiguous index
    label_to_idx = {label: idx for idx, label in enumerate(unique_topics)}

    # One-hot encode topics
    topic_features = np.zeros((len(labels), n_topics + 1))
    for i, label in enumerate(labels):
        if label == -1:
            topic_features[i, -1] = 1  # Outlier feature
        else:
            topic_features[i, label_to_idx[label]] = 1
    
    # Combine with embeddings
    features = np.hstack([embeddings, topic_features])
    
    if task == "classification":
        # Cross-validated F1
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, features, y_true, cv=5, scoring="f1_macro")
        return float(np.mean(scores))
    else:
        # Clustering ARI
        return float(adjusted_rand_score(labels, y_true))
