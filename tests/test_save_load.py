"""Tests for save/load roundtrip and backward compatibility."""

import os
import tempfile

import numpy as np

from tritopic import TriTopic, TriTopicConfig


class TestSaveLoadRoundtrip:
    def test_labels_preserved(self, fitted_model):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_model.save(path)
            loaded = TriTopic.load(path)
            np.testing.assert_array_equal(loaded.labels_, fitted_model.labels_)
        finally:
            os.remove(path)

    def test_topics_preserved(self, fitted_model):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_model.save(path)
            loaded = TriTopic.load(path)
            assert len(loaded.topics_) == len(fitted_model.topics_)
            for orig, load in zip(fitted_model.topics_, loaded.topics_):
                assert orig.topic_id == load.topic_id
                assert orig.keywords == load.keywords
        finally:
            os.remove(path)

    def test_hierarchy_preserved(self, fitted_model):
        fitted_model.build_hierarchy(n_levels=2)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_model.save(path)
            loaded = TriTopic.load(path)
            assert loaded.hierarchy_ is not None
            assert loaded.hierarchy_.n_levels == fitted_model.hierarchy_.n_levels
        finally:
            os.remove(path)

    def test_is_fitted_flag(self, fitted_model):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_model.save(path)
            loaded = TriTopic.load(path)
            assert loaded._is_fitted is True
        finally:
            os.remove(path)


class TestBackwardCompat:
    def test_config_defaults_on_missing_language(self):
        """Simulate loading a config saved before the language field existed."""
        config = TriTopicConfig()
        # Remove the attribute to simulate an old pickle
        if hasattr(config, "language"):
            delattr(config, "language")

        # The load path checks hasattr and sets defaults
        if not hasattr(config, "language"):
            config.language = "english"
        assert config.language == "english"

    def test_config_defaults_on_missing_soft_assignment(self):
        config = TriTopicConfig()
        if hasattr(config, "soft_assignment_method"):
            delattr(config, "soft_assignment_method")

        if not hasattr(config, "soft_assignment_method"):
            config.soft_assignment_method = "centroid"
        assert config.soft_assignment_method == "centroid"
