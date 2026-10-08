"""Tests for tritopic.utils.stopwords."""

from tritopic.utils.stopwords import get_stopwords, get_stopwords_set


class TestGetStopwords:
    def test_english_returns_string(self):
        result = get_stopwords("english")
        assert result == "english"

    def test_german_returns_list(self):
        result = get_stopwords("german")
        assert isinstance(result, list)
        assert "und" in result
        assert "der" in result
        assert "die" in result

    def test_french_returns_list(self):
        result = get_stopwords("french")
        assert isinstance(result, list)
        assert "et" in result

    def test_spanish_returns_list(self):
        result = get_stopwords("spanish")
        assert isinstance(result, list)
        assert "de" in result

    def test_multilingual_returns_none(self):
        result = get_stopwords("multilingual")
        assert result is None

    def test_unknown_language_returns_none(self):
        result = get_stopwords("unknown_lang")
        assert result is None

    def test_case_insensitive(self):
        assert get_stopwords("English") == "english"
        assert get_stopwords("GERMAN") is not None


class TestGetStopwordsSet:
    def test_english_returns_set_with_the(self):
        result = get_stopwords_set("english")
        assert isinstance(result, set)
        assert "the" in result

    def test_german_returns_set_with_und(self):
        result = get_stopwords_set("german")
        assert isinstance(result, set)
        assert "und" in result

    def test_multilingual_returns_empty_set(self):
        result = get_stopwords_set("multilingual")
        assert isinstance(result, set)
        assert len(result) == 0

    def test_unknown_returns_empty_set(self):
        result = get_stopwords_set("unknown_lang")
        assert isinstance(result, set)
        assert len(result) == 0
