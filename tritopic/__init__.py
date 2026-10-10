"""
TriTopic: graph-based topic modeling
=====================================

Fuses three views of a corpus into one document graph:
- semantic: sentence-transformer embeddings (reduced with UMAP)
- lexical: TF-IDF similarity
- metadata (optional): reweights edges between documents with similar metadata

and finds topics with consensus Leiden clustering. Keywords come from a
coverage-weighted c-TF-IDF; an optional LLM (``TopicInterpreter``) labels,
explains and checks the topics. Topics can be seeded from a codebook, and
``tritopic.research`` adds reliability, saturation, bridges, prevalence,
group comparisons, topic evolution, quotes and a methods report.

Basic usage:
-----------
>>> from tritopic import TriTopic
>>> model = TriTopic()
>>> labels = model.fit_transform(documents)
>>> model.get_topic_info()

Author: Roman Egger
License: MIT
"""

__version__ = "2.5.0"
__author__ = "Roman Egger"

from tritopic.core.model import TriTopic, TriTopicConfig, TopicInfo
from tritopic.core.graph_builder import GraphBuilder
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.keywords import KeywordExtractor
from tritopic.core.hierarchy import TopicNode, TopicHierarchy
from tritopic.labeling.llm_labeler import LLMLabeler, SimpleLabeler
from tritopic.labeling.interpreter import TopicInterpreter, TopicInterpretation
from tritopic.visualization.plotter import TopicVisualizer

__all__ = [
    "TriTopic",
    "TriTopicConfig",
    "TopicInfo",
    "TopicNode",
    "TopicHierarchy",
    "GraphBuilder",
    "ConsensusLeiden",
    "EmbeddingEngine",
    "KeywordExtractor",
    "LLMLabeler",
    "SimpleLabeler",
    "TopicInterpreter",
    "TopicInterpretation",
    "TopicVisualizer",
]

# Turftopic integration (optional)
try:
    from tritopic.integrations.turftopic import TriTopicModel
    __all__.append("TriTopicModel")
except ImportError:
    pass
