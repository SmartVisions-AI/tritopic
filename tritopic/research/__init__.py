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
from tritopic.research.structure import (CoassignmentResult, ResolutionLadder, codebook_coverage, coassignment,
                                         keyword_coverage, resolution_ladder, topic_cores, topic_discovery,
                                         view_composition)

__all__ = [
    "topic_reliability",
    "saturation_curve", "plot_saturation", "SaturationResult",
    "bridge_documents", "topic_connections",
    "topic_prevalence", "compare_groups", "distinctive_keywords",
    "topic_evolution", "plot_evolution", "TopicEvolution",
    "topic_quotes",
    "methods_report",
    "view_composition", "resolution_ladder", "ResolutionLadder", "topic_cores", "keyword_coverage",
    "coassignment", "CoassignmentResult", "topic_discovery", "codebook_coverage",
]
