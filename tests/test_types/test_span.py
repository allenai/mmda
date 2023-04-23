import unittest

from mmda.types import box as mmda_box
from mmda.types import span as mmda_span


class TestSpan(unittest.TestCase):
    def setUp(cls):
        cls.span = mmda_span.Span(start=0, end=0)
        cls.span_dict = {
            "start": 0,
            "end": 8,
            "box": {
                "left": 0.2,
                "top": 0.09,
                "width": 0.095,
                "height": 0.017,
                "page": 0,
            },
        }

    def test_from_json(self):
        self.assertEqual(
            self.span.from_json(self.span_dict),
            mmda_span.Span(
                start=0,
                end=8,
                box=mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
            ),
        )

    def test_to_json(self):
        self.assertEqual(self.span.from_json(self.span_dict).to_json(), self.span_dict)

    def test_is_overlap(self):
        span1 = mmda_span.Span(start=0, end=8)
        span2 = mmda_span.Span(start=0, end=8)
        self.assertTrue(span1.is_overlap(span2))

        span3 = mmda_span.Span(start=2, end=5)
        self.assertTrue(span1.is_overlap(span3))

        span4 = mmda_span.Span(start=8, end=10)
        self.assertFalse(span1.is_overlap(span4))

        span5 = mmda_span.Span(start=10, end=12)
        self.assertFalse(span1.is_overlap(span5))

    def small_spans_to_big_span(self):
        spans = [
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=8, end=16),
            mmda_span.Span(start=16, end=24),
        ]
        self.assertEqual(
            self.span.small_spans_to_big_span(spans),
            mmda_span.Span(start=0, end=24),
        )

    def test_cluster_spans(self):
        # overlapping spans
        spans = [
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=0, end=8),
        ]
        self.assertListEqual(mmda_span.Span.cluster_spans(spans), [[0, 1, 2]])

        # non-overlapping spans
        spans = [
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=8, end=16),
            mmda_span.Span(start=16, end=24),
        ]
        self.assertListEqual(mmda_span.Span.cluster_spans(spans), [[0], [1], [2]])

        # partially overlapping spans
        spans = [
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=9, end=16),
            mmda_span.Span(start=10, end=15),
        ]
        self.assertListEqual(mmda_span.Span.cluster_spans(spans), [[0], [1, 2]])
