import unittest
from mmda.types import box as mmda_box


class TestBox(unittest.TestCase):
    def setUp(cls) -> None:
        cls.box_dict = {'left': 0.2,
                        'top': 0.09,
                        'width': 0.095,
                        'height': 0.017,
                        'page': 0}
        cls.box = mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0)

    def test_from_json(self):
        self.assertEqual(self.box.from_json(self.box_dict), self.box)

    def test_to_json(self):
        self.assertEqual(self.box.to_json(), self.box_dict)
