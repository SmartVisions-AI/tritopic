"""Tests for soft assignment features (get_document_topics, topic_overlap_matrix)."""

import numpy as np
import pandas as pd


class TestGetDocumentTopics:
    def test_returns_list_of_tuples(self, fitted_model):
        result = fitted_model.get_document_topics(0, top_n=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

    def test_probabilities_sum_approx_one(self, fitted_model):
        n_topics = len([t for t in fitted_model.topics_ if t.topic_id != -1])
        result = fitted_model.get_document_topics(0, top_n=n_topics)
        total = sum(prob for _, prob in result)
        assert abs(total - 1.0) < 0.05

    def test_sorted_descending(self, fitted_model):
        result = fitted_model.get_document_topics(0, top_n=3)
        probs = [p for _, p in result]
        assert probs == sorted(probs, reverse=True)


class TestTopicOverlapMatrix:
    def test_returns_dataframe(self, fitted_model):
        overlap = fitted_model.topic_overlap_matrix(threshold=0.1)
        assert isinstance(overlap, pd.DataFrame)

    def test_symmetric(self, fitted_model):
        overlap = fitted_model.topic_overlap_matrix(threshold=0.1)
        np.testing.assert_array_equal(overlap.values, overlap.values.T)

    def test_diagonal_positive(self, fitted_model):
        overlap = fitted_model.topic_overlap_matrix(threshold=0.1)
        diag = np.diag(overlap.values)
        assert all(d > 0 for d in diag)

    def test_column_order_matches_topics(self, fitted_model):
        overlap = fitted_model.topic_overlap_matrix(threshold=0.1)
        expected_ids = [t.topic_id for t in fitted_model.topics_ if t.topic_id != -1]
        expected_cols = [f"Topic {tid}" for tid in expected_ids]
        assert list(overlap.columns) == expected_cols
