"""Basic tests for the TriTopic model fitting pipeline."""

import numpy as np
import pandas as pd

from tritopic import TriTopic
from tritopic.core.hierarchy import TopicHierarchy


class TestFitTransform:
    def test_returns_ndarray(self, fitted_model):
        assert isinstance(fitted_model.labels_, np.ndarray)

    def test_labels_length(self, fitted_model, fake_documents):
        assert len(fitted_model.labels_) == len(fake_documents)

    def test_topics_not_empty(self, fitted_model):
        assert len(fitted_model.topics_) > 0

    def test_probabilities_shape(self, fitted_model, fake_documents):
        n_topics = len([t for t in fitted_model.topics_ if t.topic_id != -1])
        assert fitted_model.probabilities_ is not None
        assert fitted_model.probabilities_.shape == (len(fake_documents), n_topics)

    def test_get_topic_info_returns_dataframe(self, fitted_model):
        df = fitted_model.get_topic_info()
        assert isinstance(df, pd.DataFrame)
        assert "Topic" in df.columns
        assert "Size" in df.columns
        assert "Keywords" in df.columns


class TestBuildHierarchy:
    def test_returns_hierarchy(self, fitted_model):
        h = fitted_model.build_hierarchy(n_levels=2)
        assert isinstance(h, TopicHierarchy)

    def test_correct_n_levels(self, fitted_model):
        h = fitted_model.build_hierarchy(n_levels=2)
        assert h.n_levels == 2

    def test_stored_on_model(self, fitted_model):
        fitted_model.build_hierarchy(n_levels=2)
        assert fitted_model.hierarchy_ is not None


class TestDivide:
    def test_returns_list_of_topicinfo(self, fitted_model):
        non_outlier = [t for t in fitted_model.topics_ if t.topic_id != -1]
        if not non_outlier:
            return
        tid = non_outlier[0].topic_id
        subtopics = fitted_model.divide(topic_id=tid, n_subtopics=2)
        assert isinstance(subtopics, list)
        assert len(subtopics) >= 1
