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

    def test_are_disjoint(self):
        self.assertTrue(
            mmda_span.Span.are_disjoint(
                spans=[mmda_span.Span(start=0, end=1), mmda_span.Span(start=1, end=2)]
            )
        )
        self.assertFalse(
            mmda_span.Span.are_disjoint(
                spans=[mmda_span.Span(start=0, end=2), mmda_span.Span(start=1, end=3)]
            )
        )
