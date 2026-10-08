"""
Hierarchical Topic Organization for TriTopic
===============================================

Multi-resolution topic hierarchy built from Leiden community detection
at different resolution levels.  Levels are linked via majority-vote
assignment of documents to parent nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopicNode:
    """A single node in the topic hierarchy tree.

    Attributes
    ----------
    node_id : str
        Unique identifier, e.g. ``"L0_3"`` (level 0, topic 3).
    level : int
        Depth in the hierarchy (0 = coarsest).
    topic_id : int
        Local topic id at this level.
    size : int
        Number of documents assigned to this node.
    keywords : list[str]
        Top keywords for this node.
    keyword_scores : list[float]
        Scores corresponding to *keywords*.
    doc_indices : np.ndarray
        Document indices belonging to this node.
    centroid : np.ndarray | None
        Mean embedding of member documents.
    parent : TopicNode | None
        Parent node (``None`` for roots).
    children : list[TopicNode]
        Child nodes (empty for leaves).
    label : str | None
        Optional human-readable label.
    """

    node_id: str
    level: int
    topic_id: int
    size: int
    keywords: list[str] = field(default_factory=list)
    keyword_scores: list[float] = field(default_factory=list)
    doc_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    centroid: np.ndarray | None = field(default=None, repr=False)
    parent: TopicNode | None = field(default=None, repr=False)
    children: list[TopicNode] = field(default_factory=list, repr=False)
    label: str | None = None

    # ------------------------------------------------------------------
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_subtopics(self, depth: int = 1) -> list[TopicNode]:
        """Return subtopics up to *depth* levels below this node."""
        if depth <= 0 or self.is_leaf():
            return []
        result = list(self.children)
        if depth > 1:
            for child in self.children:
                result.extend(child.get_subtopics(depth - 1))
        return result

    def flatten(self) -> list[TopicNode]:
        """Return this node and all descendants as a flat list."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes

    def __repr__(self) -> str:
        lbl = self.label or ", ".join(self.keywords[:3])
        return f"TopicNode({self.node_id}, size={self.size}, '{lbl}')"


@dataclass
class TopicHierarchy:
    """Container for the full multi-resolution topic hierarchy.

    Attributes
    ----------
    roots : list[TopicNode]
        Top-level (coarsest) topic nodes.
    levels : list[list[TopicNode]]
        All nodes grouped by level, ``levels[0]`` = coarsest.
    resolution_levels : list[float]
        Leiden resolution used at each level.
    """

    roots: list[TopicNode] = field(default_factory=list)
    levels: list[list[TopicNode]] = field(default_factory=list)
    resolution_levels: list[float] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def cut(self, depth: int) -> list[TopicNode]:
        """Return all nodes at a given *depth* level."""
        if depth < 0 or depth >= self.n_levels:
            raise IndexError(f"depth {depth} out of range (0..{self.n_levels - 1})")
        return list(self.levels[depth])

    def flatten(self) -> list[TopicNode]:
        """Return every node in the hierarchy."""
        return [node for level in self.levels for node in level]

    def get_node(self, node_id: str) -> TopicNode | None:
        """Look up a node by its ``node_id``."""
        for node in self.flatten():
            if node.node_id == node_id:
                return node
        return None

    def __repr__(self) -> str:
        level_sizes = [len(lvl) for lvl in self.levels]
        return f"TopicHierarchy(levels={self.n_levels}, topics_per_level={level_sizes})"
