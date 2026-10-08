"""
TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement
===================================================================

A state-of-the-art topic modeling library that combines:
- Semantic embeddings (Sentence-BERT, Instructor, BGE)
- Lexical similarity (BM25)
- Metadata context (optional)

With advanced techniques:
- Leiden clustering with consensus
- Mutual kNN + SNN graph construction
- Iterative refinement loop
- LLM-powered topic labeling

Basic usage:
-----------
>>> from tritopic import TriTopic
>>> model = TriTopic()
>>> topics = model.fit_transform(documents)
>>> model.visualize()

Author: Roman Egger
License: MIT
"""

__version__ = "2.4.0"
__author__ = "Roman Egger"

from tritopic.core.model import TriTopic, TriTopicConfig, TopicInfo
from tritopic.core.graph_builder import GraphBuilder
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.keywords import KeywordExtractor
from tritopic.core.hierarchy import TopicNode, TopicHierarchy
from tritopic.labeling.llm_labeler import LLMLabeler, SimpleLabeler
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
    "TopicVisualizer",
]

# Turftopic integration (optional)
try:
    from tritopic.integrations.turftopic import TriTopicModel
    __all__.append("TriTopicModel")
except ImportError:
    pass
