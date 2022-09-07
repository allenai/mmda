"""


@kylel

"""


import unittest

from mmda.types.span import Span
from mmda.types.box import Box
from mmda.utils.tools import merge_neighbor_spans


class TestMergeNeighborSpans(unittest.TestCase):
    def test_merge_multiple_neighbor_spans(self):
        spans = [Span(start=0, end=10), Span(start=11, end=20), Span(start=21, end=30)]
        out = merge_neighbor_spans(spans=spans, distance=1)
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 30

    def test_different_distances(self):
        spans = [Span(start=0, end=10), Span(start=15, end=20)]
        out = merge_neighbor_spans(spans=spans, distance=1)
        assert out == spans  # no merge happened

        out = merge_neighbor_spans(spans=spans, distance=2)
        assert out == spans  # no merge happened

        out = merge_neighbor_spans(spans=spans, distance=4)
        assert out == spans  # no merge happened

        out = merge_neighbor_spans(spans=spans, distance=5)
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 20

    def test_zero_distance(self):
        spans = [Span(start=0, end=10), Span(start=10, end=20)]
        out = merge_neighbor_spans(spans=spans, distance=0)
        assert len(out) == 1
        assert isinstance(out[0], Span)
        assert out[0].start == 0
        assert out[0].end == 20

    def test_handling_of_boxes(self):
        spans = [
            Span(start=0, end=10, box=Box(l=0, t=0, w=1, h=1, page=0)),
            Span(start=11, end=20, box=Box(l=1, t=1, w=2, h=2, page=1)),
            Span(start=100, end=150, box=Box(l=2, t=2, w=3, h=3, page=2))
        ]
        out = merge_neighbor_spans(spans=spans, distance=1)
        assert len(out) == 2
        assert isinstance(out[0], Span)
        assert isinstance(out[1], Span)
        assert out[0].start == 0
        assert out[0].end == 20
        assert out[1].start == 100
        assert out[1].end == 150
        assert out[0].box is None
        assert out[1].box == spans[-1].box  # unmerged spans keep their original box
