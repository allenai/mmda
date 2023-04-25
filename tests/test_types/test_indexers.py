import unittest

from mmda.types import Box, BoxGroup, Span, SpanGroup
from mmda.types.indexers import BoxGroupIndexer, SpanGroupIndexer


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


class TestBoxGroupIndexer(unittest.TestCase):
    def test_overlap_within_single_boxgroup_fails_checks(self):
        box_groups = [
            BoxGroup(
                id=1,
                boxes=[Box(0, 0, 5, 5, page=0), Box(4, 4, 7, 7, page=0)],
                allow_overlap=True,
            )
        ]

        with self.assertRaises(ValueError):
            BoxGroupIndexer(box_groups)

    def test_overlap_between_boxgroups_fails_checks(self):
        box_groups = [
            BoxGroup(
                id=1, boxes=[Box(0, 0, 5, 5, page=0), Box(5.01, 5.01, 8, 8, page=0)]
            ),
            BoxGroup(id=2, boxes=[Box(6, 6, 10, 10, page=0)]),
        ]

        with self.assertRaises(ValueError):
            BoxGroupIndexer(box_groups)

    def test_finds_matching_groups_in_doc_order(self):
        box_groups_to_index = [
            BoxGroup(id=1, boxes=[Box(0, 0, 1, 1, page=0), Box(2, 2, 1, 1, page=0)]),
            BoxGroup(id=2, boxes=[Box(4, 4, 1, 1, page=0)]),
            BoxGroup(id=3, boxes=[Box(100, 100, 1, 1, page=0)]),
        ]

        index = BoxGroupIndexer(box_groups_to_index)

        # should intersect 1 and 2 but not 3
        probe = BoxGroup(id=4, boxes=[Box(1, 1, 5, 5, page=0), Box(9, 9, 5, 5, page=0)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [box_groups_to_index[0], box_groups_to_index[1]])
