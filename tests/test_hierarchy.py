"""Tests for tritopic.core.hierarchy — TopicNode and TopicHierarchy."""

import numpy as np
import pytest

from tritopic.core.hierarchy import TopicNode, TopicHierarchy


def _make_node(node_id, level=0, topic_id=0, size=10):
    return TopicNode(
        node_id=node_id,
        level=level,
        topic_id=topic_id,
        size=size,
        keywords=["kw1", "kw2"],
        keyword_scores=[0.5, 0.3],
        doc_indices=np.arange(size),
    )


class TestTopicNode:
    def test_is_leaf_true(self):
        node = _make_node("L0_0")
        assert node.is_leaf() is True

    def test_is_leaf_false(self):
        parent = _make_node("L0_0")
        child = _make_node("L1_0", level=1)
        parent.children.append(child)
        assert parent.is_leaf() is False

    def test_get_subtopics_depth_1(self):
        parent = _make_node("L0_0")
        c1 = _make_node("L1_0", level=1)
        c2 = _make_node("L1_1", level=1)
        parent.children = [c1, c2]
        subs = parent.get_subtopics(depth=1)
        assert len(subs) == 2
        assert c1 in subs
        assert c2 in subs

    def test_get_subtopics_depth_2(self):
        root = _make_node("L0_0")
        child = _make_node("L1_0", level=1)
        grandchild = _make_node("L2_0", level=2)
        root.children = [child]
        child.children = [grandchild]
        subs = root.get_subtopics(depth=2)
        assert grandchild in subs
        assert child in subs

    def test_get_subtopics_depth_0(self):
        node = _make_node("L0_0")
        node.children = [_make_node("L1_0", level=1)]
        assert node.get_subtopics(depth=0) == []

    def test_flatten(self):
        root = _make_node("L0_0")
        child = _make_node("L1_0", level=1)
        grandchild = _make_node("L2_0", level=2)
        root.children = [child]
        child.children = [grandchild]
        flat = root.flatten()
        assert len(flat) == 3
        assert flat[0] is root
        assert flat[1] is child
        assert flat[2] is grandchild


class TestTopicHierarchy:
    def _build_hierarchy(self):
        l0 = [_make_node("L0_0"), _make_node("L0_1")]
        l1 = [_make_node("L1_0", level=1), _make_node("L1_1", level=1),
              _make_node("L1_2", level=1)]
        l0[0].children = [l1[0], l1[1]]
        l0[1].children = [l1[2]]
        for c in l1[:2]:
            c.parent = l0[0]
        l1[2].parent = l0[1]
        return TopicHierarchy(
            roots=l0,
            levels=[l0, l1],
            resolution_levels=[0.25, 1.0],
        )

    def test_n_levels(self):
        h = self._build_hierarchy()
        assert h.n_levels == 2

    def test_cut_depth_0(self):
        h = self._build_hierarchy()
        nodes = h.cut(0)
        assert len(nodes) == 2

    def test_cut_depth_1(self):
        h = self._build_hierarchy()
        nodes = h.cut(1)
        assert len(nodes) == 3

    def test_cut_out_of_range(self):
        h = self._build_hierarchy()
        with pytest.raises(IndexError):
            h.cut(5)

    def test_cut_negative(self):
        h = self._build_hierarchy()
        with pytest.raises(IndexError):
            h.cut(-1)

    def test_flatten(self):
        h = self._build_hierarchy()
        flat = h.flatten()
        assert len(flat) == 5

    def test_get_node_found(self):
        h = self._build_hierarchy()
        node = h.get_node("L1_2")
        assert node is not None
        assert node.node_id == "L1_2"

    def test_get_node_not_found(self):
        h = self._build_hierarchy()
        assert h.get_node("nonexistent") is None

    def test_parent_child_linking(self):
        h = self._build_hierarchy()
        child = h.get_node("L1_0")
        assert child.parent is not None
        assert child.parent.node_id == "L0_0"
        assert child in child.parent.children
