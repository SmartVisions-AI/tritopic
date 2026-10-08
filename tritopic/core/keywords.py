"""
Keyword Extraction for TriTopic
================================

Extract representative keywords for topics using:
- c-TF-IDF (class-based TF-IDF, like BERTopic)
- BM25 scoring
- KeyBERT (embedding-based)
"""

from __future__ import annotations

from typing import Literal
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from tritopic.utils.stopwords import TOKEN_PATTERN, get_stopwords


class KeywordExtractor:
    """
    Extract keywords for topics.

    Supports multiple extraction methods for flexibility.

    The corpus document-term matrix is computed once and cached, so
    extracting keywords for all topics (and refreshing them after
    ``reduce_outliers`` / ``reduce_topics`` / ``merge_topics``) does not
    re-tokenize the corpus.

    Parameters
    ----------
    method : str
        Extraction method: "ctfidf", "bm25", or "keybert"
    n_keywords : int
        Number of keywords to extract per topic. Default: 10
    ngram_range : tuple
        N-gram range for keyword extraction. Default: (1, 2)
    language : str
        Stopword language ("english", "german", "french", "spanish",
        "multilingual"). Default: "english"
    """

    def __init__(
        self,
        method: Literal["ctfidf", "bm25", "keybert"] = "ctfidf",
        n_keywords: int = 10,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        language: str = "english",
    ):
        self.method = method
        self.n_keywords = n_keywords
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.language = language

        self._vectorizer = None
        self._vocabulary = None
        self._idf = None
        self._keybert_model = None
        self._doc_term = None   # (n_docs, n_terms) counts for the fitted corpus
        self._corpus_key = None
        self._bm25_corpus_mean = None

    def extract(
        self,
        topic_docs: list[str],
        all_docs: list[str] | None = None,
        n_keywords: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords from topic documents.

        Parameters
        ----------
        topic_docs : list[str]
            Documents belonging to the topic.
        all_docs : list[str], optional
            All documents in corpus (needed for c-TF-IDF).
        n_keywords : int, optional
            Override default n_keywords.

        Returns
        -------
        keywords : list[str]
            Top keywords for the topic.
        scores : list[float]
            Keyword scores.
        """
        n = n_keywords or self.n_keywords

        if self.method == "keybert":
            return self._extract_keybert(topic_docs, n)
        if self.method not in ("ctfidf", "bm25"):
            raise ValueError(f"Unknown method: {self.method}")

        self._fit_corpus(all_docs or topic_docs)
        topic_tf = self._vectorizer.transform(topic_docs)
        if self.method == "ctfidf":
            scores = self._ctfidf_scores(
                np.asarray(topic_tf.sum(axis=0)).ravel(),
                np.bincount(topic_tf.tocsr().indices, minlength=topic_tf.shape[1]),
                topic_tf.shape[0],
            )
        else:
            scores = self._bm25_scores(topic_tf)
        return self._top_n(scores, n)

    def reset(self) -> None:
        """Reset the fitted vectorizer state. Call before re-fitting on new data."""
        self._vectorizer = None
        self._vocabulary = None
        self._idf = None
        self._keybert_model = None
        self._doc_term = None
        self._corpus_key = None
        self._bm25_corpus_mean = None

    def fit_corpus(self, all_docs: list[str]) -> "csr_matrix":
        """Fit the vectorizer on the corpus and return the cached
        document-term count matrix (also reused by the lexical view)."""
        self._fit_corpus(all_docs)
        return self._doc_term

    def _fit_corpus(self, all_docs: list[str]) -> None:
        """Fit the vectorizer and cache the corpus document-term matrix."""
        key = (id(all_docs), len(all_docs))
        if self._doc_term is not None and self._corpus_key == key:
            return

        self._vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=get_stopwords(self.language),
            token_pattern=TOKEN_PATTERN,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self._doc_term = self._vectorizer.fit_transform(all_docs).tocsr()
        self._vocabulary = self._vectorizer.get_feature_names_out()
        # Per-term document frequency
        doc_freq = np.bincount(self._doc_term.indices, minlength=len(self._vocabulary))
        self._idf = np.log(len(all_docs) / (1 + doc_freq))
        self._corpus_key = key
        self._bm25_corpus_mean = None

    def _ctfidf_scores(
        self,
        topic_tf: np.ndarray,
        topic_df: np.ndarray,
        n_topic_docs: int,
    ) -> np.ndarray:
        """
        Class-based TF-IDF with document coverage.

        score(t) = IDF(t) * sqrt( tf_share(t) * coverage(t) )

        * ``tf_share``: share of the topic's tokens that are *t* (classic
          c-TF-IDF term frequency of the concatenated "class document").
        * ``coverage``: fraction of the topic's documents containing *t*.

        Term frequency alone favours words repeated in a few long documents;
        coverage alone favours boilerplate.  Their geometric mean keeps terms
        that are both frequent and spread across the topic, which raised NPMI
        coherence by ~45% on the dev benchmarks (20NG, BBC, AG News, arXiv)
        while keeping keywords specific.
        """
        tf_share = topic_tf / (topic_tf.sum() + 1e-10)
        coverage = topic_df / max(n_topic_docs, 1)
        return np.sqrt(tf_share * coverage) * self._idf

    def _bm25_weights(self, tf_matrix) -> "csr_matrix":
        """BM25 term weights (Okapi, k1=1.5, b=0.75) using corpus statistics."""
        k1, b = 1.5, 0.75
        X = self._doc_term
        n_docs = X.shape[0]
        avgdl = X.sum() / max(n_docs, 1)
        doc_freq = np.bincount(X.indices, minlength=X.shape[1])
        idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        idf = np.where(idf < 0, 0.25 * idf[idf > 0].mean() if (idf > 0).any() else 0, idf)

        W = tf_matrix.tocsr().astype(float, copy=True)
        dl = np.asarray(W.sum(axis=1)).ravel()
        row_of_entry = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        tf = W.data
        W.data = idf[W.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[row_of_entry] / avgdl))
        return W

    def _bm25_scores(self, topic_tf) -> np.ndarray:
        """
        Topic-specificity: average BM25 relevance of each term within the
        topic's documents vs. across the whole corpus, times log(1 + freq).
        """
        if self._bm25_corpus_mean is None:
            self._bm25_corpus_mean = np.asarray(
                self._bm25_weights(self._doc_term).mean(axis=0)
            ).ravel()
        topic_mean = np.asarray(self._bm25_weights(topic_tf).mean(axis=0)).ravel()
        freq = np.asarray(topic_tf.sum(axis=0)).ravel()
        scores = topic_mean / (self._bm25_corpus_mean + 1e-10) * np.log1p(freq)
        max_score = scores.max() if scores.size else 0
        return scores / max_score if max_score > 0 else scores

    def _top_n(self, scores: np.ndarray, n: int) -> tuple[list[str], list[float]]:
        """Top-n terms with a positive score, best first."""
        n = min(n, scores.size)
        if n == 0:
            return [], []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        top = top[scores[top] > 0]
        return [self._vocabulary[i] for i in top], [float(scores[i]) for i in top]

    def _extract_keybert(
        self,
        topic_docs: list[str],
        n_keywords: int,
    ) -> tuple[list[str], list[float]]:
        """
        Extract keywords using KeyBERT (embedding-based).

        KeyBERT finds keywords by comparing candidate embeddings
        to the document embedding.  The model is cached across calls.
        """
        from keybert import KeyBERT

        # Concatenate topic docs
        topic_text = " ".join(topic_docs)

        # Reuse cached KeyBERT model
        if self._keybert_model is None:
            self._keybert_model = KeyBERT()

        # Extract keywords
        keywords_with_scores = self._keybert_model.extract_keywords(
            topic_text,
            keyphrase_ngram_range=self.ngram_range,
            stop_words=get_stopwords(self.language),
            top_n=n_keywords,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.5,
        )

        keywords = [kw for kw, score in keywords_with_scores]
        scores = [float(score) for kw, score in keywords_with_scores]

        return keywords, scores

    def extract_all_topics(
        self,
        documents: list[str],
        labels: np.ndarray,
        n_keywords: int | None = None,
        include_outliers: bool = False,
        method: str | None = None,
    ) -> dict[int, tuple[list[str], list[float]]]:
        """
        Extract keywords for all topics at once.

        Parameters
        ----------
        documents : list[str]
            All documents.
        labels : np.ndarray
            Topic assignments.
        n_keywords : int, optional
            Override default n_keywords.
        include_outliers : bool
            Also extract keywords for the outlier group (-1).
        method : str, optional
            Override ``self.method`` for this call.

        Returns
        -------
        topic_keywords : dict
            Mapping from topic_id to (keywords, scores).
        """
        n = n_keywords or self.n_keywords
        labels = np.asarray(labels)
        topic_ids = [int(t) for t in np.unique(labels) if include_outliers or t != -1]

        method = method or self.method
        if method == "keybert":
            return {
                t: self._extract_keybert([documents[i] for i in np.where(labels == t)[0]], n)
                for t in topic_ids
            }
        if method not in ("ctfidf", "bm25"):
            raise ValueError(f"Unknown method: {method}")

        self._fit_corpus(documents)
        result = {}
        if method == "ctfidf":
            # Class term frequencies for all topics in one sparse product
            ids = np.array(topic_ids)
            row_of_doc = np.searchsorted(ids, labels)
            valid = np.isin(labels, ids)
            indicator = csr_matrix(
                (np.ones(valid.sum()), (row_of_doc[valid], np.where(valid)[0])),
                shape=(len(ids), len(labels)),
            )
            class_tf = (indicator @ self._doc_term).toarray()
            presence = self._doc_term.copy()
            presence.data = np.ones_like(presence.data)
            class_df = (indicator @ presence).toarray()
            class_sizes = np.asarray(indicator.sum(axis=1)).ravel()
            for row, t in enumerate(topic_ids):
                result[t] = self._top_n(
                    self._ctfidf_scores(class_tf[row], class_df[row], class_sizes[row]), n
                )
        else:
            for t in topic_ids:
                result[t] = self._top_n(self._bm25_scores(self._doc_term[labels == t]), n)
        return result


class KeyphraseExtractor:
    """
    Extract keyphrases (multi-word) using YAKE or TextRank.
    """
    
    def __init__(
        self,
        method: Literal["yake", "textrank"] = "yake",
        n_keyphrases: int = 10,
        max_ngram: int = 3,
    ):
        self.method = method
        self.n_keyphrases = n_keyphrases
        self.max_ngram = max_ngram
    
    def extract(self, text: str) -> list[tuple[str, float]]:
        """Extract keyphrases from text."""
        if self.method == "yake":
            return self._extract_yake(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _extract_yake(self, text: str) -> list[tuple[str, float]]:
        """Extract using YAKE algorithm."""
        try:
            import yake
        except ImportError:
            # Fallback to simple extraction
            return self._simple_extract(text)
        
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=self.max_ngram,
            dedupLim=0.7,
            top=self.n_keyphrases,
            features=None,
        )
        
        keywords = kw_extractor.extract_keywords(text)
        
        # YAKE returns (keyword, score) where lower score is better
        # Invert for consistency
        max_score = max(s for _, s in keywords) if keywords else 1
        return [(kw, 1 - s/max_score) for kw, s in keywords]
    
    def _simple_extract(self, text: str) -> list[tuple[str, float]]:
        """Simple n-gram frequency extraction."""
        import re
        from collections import Counter
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Generate n-grams
        ngrams = []
        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.append(ngram)
        
        # Count and return top
        counts = Counter(ngrams)
        top = counts.most_common(self.n_keyphrases)
        
        max_count = top[0][1] if top else 1
        return [(phrase, count/max_count) for phrase, count in top]
