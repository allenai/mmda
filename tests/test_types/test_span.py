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
        span = mmda_span.Span(start=0, end=2)
        self.assertTrue(span.is_overlap(mmda_span.Span(start=0, end=1)))
        self.assertTrue(span.is_overlap(mmda_span.Span(start=1, end=2)))
        self.assertFalse(span.is_overlap(mmda_span.Span(start=2, end=3)))
        self.assertFalse(span.is_overlap(mmda_span.Span(start=4, end=5)))

    def test_small_spans_to_big_span(self):
        spans = [
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=8, end=16),
            mmda_span.Span(start=16, end=24),
        ]
        self.assertEqual(
            mmda_span.Span.small_spans_to_big_span(spans=spans, merge_boxes=False),
            mmda_span.Span(start=0, end=24),
        )
        # if no boxes, should still work
        self.assertEqual(
            mmda_span.Span.small_spans_to_big_span(spans=spans, merge_boxes=True),
            mmda_span.Span(start=0, end=24),
        )

    def test_small_spans_to_big_span_unsorted(self):
        spans = [
            mmda_span.Span(start=8, end=16),
            mmda_span.Span(start=0, end=8),
            mmda_span.Span(start=16, end=24),
        ]
        self.assertEqual(
            mmda_span.Span.small_spans_to_big_span(spans=spans),
            mmda_span.Span(start=0, end=24),
        )

        spans = [
            mmda_span.Span(start=16, end=24),
            mmda_span.Span(start=8, end=16),
            mmda_span.Span(start=0, end=8),
        ]
        self.assertEqual(
            mmda_span.Span.small_spans_to_big_span(spans=spans),
            mmda_span.Span(start=0, end=24),
        )

    def test_are_disjoint(self):
        # should be disjoint
        span1 = mmda_span.Span(start=0, end=1)
        span2 = mmda_span.Span(start=1, end=2)
        self.assertTrue(mmda_span.Span.are_disjoint(spans=[span1, span2]))
        self.assertTrue(mmda_span.Span.are_disjoint(spans=[span2, span1]))

        # should overlap
        span3 = mmda_span.Span(start=0, end=2)
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span1, span3]))
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span3, span1]))

        # should handle strict containment
        span4 = mmda_span.Span(start=0, end=3)
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span1, span4]))
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span4, span1]))
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span2, span4]))
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span4, span2]))

        # should handle exact equality
        span5 = mmda_span.Span(start=0, end=1)
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span1, span5]))
        self.assertFalse(mmda_span.Span.are_disjoint(spans=[span5, span1]))
