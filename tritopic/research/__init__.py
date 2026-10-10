"""
Research toolkit: outputs for analysing and reporting a fitted TriTopic model.

>>> from tritopic.research import topic_reliability, saturation_curve, bridge_documents
"""

from tritopic.research.bridges import bridge_documents, topic_connections
from tritopic.research.evolution import TopicEvolution, plot_evolution, topic_evolution
from tritopic.research.prevalence import compare_groups, distinctive_keywords, topic_prevalence
from tritopic.research.quotes import topic_quotes
from tritopic.research.reliability import topic_reliability
from tritopic.research.report import methods_report
from tritopic.research.saturation import SaturationResult, plot_saturation, saturation_curve

__all__ = [
    "topic_reliability",
    "saturation_curve", "plot_saturation", "SaturationResult",
    "bridge_documents", "topic_connections",
    "topic_prevalence", "compare_groups", "distinctive_keywords",
    "topic_evolution", "plot_evolution", "TopicEvolution",
    "topic_quotes",
    "methods_report",
]
