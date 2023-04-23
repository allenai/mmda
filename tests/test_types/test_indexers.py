import unittest

from mmda.types import Span, SpanGroup
from mmda.types.indexers import SpanGroupIndexer


class TestSpanGroupIndexer(unittest.TestCase):
    def test_overlap_within_single_spangroup_fails_checks(self):
        span_groups = [
            SpanGroup(id=1, spans=[Span(0, 5), Span(4, 7)], allow_overlap=True)
        ]

        with self.assertRaises(ValueError):
            SpanGroupIndexer(span_groups)

    def test_overlap_between_spangroups_fails_checks(self):
        span_groups = [
            SpanGroup(id=1, spans=[Span(0, 5), Span(5, 8)]),
            SpanGroup(id=2, spans=[Span(6, 10)]),
        ]

        with self.assertRaises(ValueError):
            SpanGroupIndexer(span_groups)

    def test_finds_matching_groups_in_doc_order(self):
        span_groups_to_index = [
            SpanGroup(id=1, spans=[Span(0, 5), Span(5, 8)]),
            SpanGroup(id=2, spans=[Span(9, 10)]),
            SpanGroup(id=3, spans=[Span(100, 105)]),
        ]

        index = SpanGroupIndexer(span_groups_to_index)

        # should intersect 1 and 2 but not 3
        probe = SpanGroup(id=3, spans=[Span(1, 7), Span(9, 20)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [span_groups_to_index[0], span_groups_to_index[1]])
