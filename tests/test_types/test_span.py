import unittest
from mmda.types import span as mmda_span
from mmda.types import box as mmda_box


class TestBox(unittest.TestCase):
    def setUp(cls):
        cls.span = mmda_span.Span(start=0, end=0)
        cls.span_dict = {'start': 0,
                         'end': 8,
                         'box': {'left': 0.2,
                                 'top': 0.09,
                                 'width': 0.095,
                                 'height': 0.017,
                                 'page': 0}}

    def test_from_json(self):
        self.assertEqual(self.span.from_json(self.span_dict),
                         mmda_span.Span(start=0, end=8, box=mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0)))

    def test_to_json(self):
        self.assertEqual(self.span.from_json(self.span_dict).to_json(),
                         self.span_dict)

