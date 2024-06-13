import unittest

from mmda.types import box as mmda_box


class TestBox(unittest.TestCase):
    def setUp(cls) -> None:
        cls.box_dict = {
            "left": 0.2,
            "top": 0.09,
            "width": 0.095,
            "height": 0.017,
            "page": 0,
        }
        cls.box = mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0)

    def test_to_from_json(self):
        box = self.box.from_json(self.box_dict)
        self.assertDictEqual(box.to_json(), self.box_dict)

    def test_cluster_boxes(self):
        # overlapping boxes
        boxes = [
            mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
        ]
        self.assertListEqual(mmda_box.Box.cluster_boxes(boxes), [[0, 1, 2]])

        # on-overlapping boxes
        boxes = [
            mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.3, t=0.20, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.4, t=0.30, w=0.095, h=0.017, page=0),
        ]
        self.assertListEqual(mmda_box.Box.cluster_boxes(boxes), [[0], [1], [2]])

        # partially overlapping boxes
        boxes = [
            mmda_box.Box(l=0.2, t=0.09, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.3, t=0.20, w=0.095, h=0.017, page=0),
            mmda_box.Box(l=0.301, t=0.201, w=0.095, h=0.017, page=0),
        ]
        self.assertListEqual(mmda_box.Box.cluster_boxes(boxes), [[0], [1, 2]])

    def test_create_invalid_box(self):
        box = mmda_box.Box(l=0.7, t=0.4, w=0.3, h=0.6, page=0)
        with self.assertRaises(ValueError):
            box = mmda_box.Box(l=0.7 + 0.0000001, t=0.2, w=0.3, h=0.4, page=0)
            box = mmda_box.Box(l=0.7, t=0.4, w=0.3, h=0.6 + 0.000001, page=0)
        with self.assertRaises(ValueError):
            box = mmda_box.Box(l=0.7, t=0.4, w=0.3, h=0.6, page=-1)

    def test_shrink(self):
        # usual
        box = mmda_box.Box(l=0.1, t=0.2, w=0.3, h=0.4, page=0)
        box.shrink(delta=0.1)
        self.assertAlmostEqual(box.l, 0.2)  # 0.1 + 0.1
        self.assertAlmostEqual(box.t, 0.3)  # 0.2 + 0.1
        self.assertAlmostEqual(box.w, 0.1)  # 0.3 - 0.1 * 2
        self.assertAlmostEqual(box.h, 0.2)  # 0.4 - 0.1 * 2

        # shrinking until inverts box. would ignore shrinking along appropriate axis.
        box = mmda_box.Box(l=0.9, t=0.5, w=0.1, h=0.3, page=0)
        box.shrink(delta=0.1, ignore=True)
        self.assertAlmostEqual(box.l, 0.9)  # ignored
        self.assertAlmostEqual(box.t, 0.6)  # adjusted; 0.5 + 0.1
        self.assertAlmostEqual(box.w, 0.1)  # ignored
        self.assertAlmostEqual(box.h, 0.1)  # adjusted; 0.3 - 2 * 0.1

        # shrinking until out of bounds. would clip along appropriate axis.
        # actually... does this ever happen unless Box is already out of bounds?

    def test_cluster_boxes_hard(self):
        # from 4be952924cd565488b4a239dc6549095029ee578.pdf, page 2, tokens 650:655
        boxes = [
            mmda_box.Box(
                l=0.7761069934640523,
                t=0.14276190217171716,
                w=0.005533858823529373,
                h=0.008037272727272593,
                page=2,
            ),
            mmda_box.Box(
                l=0.7836408522875816,
                t=0.14691867138383832,
                w=0.005239432156862763,
                h=0.005360666666666692,
                page=2,
            ),
            mmda_box.Box(
                l=1.001, t=0.3424244465151515, w=-0.002, h=0.008037272727272737, page=2
            ),
            mmda_box.Box(
                l=1.001, t=0.3424244465151515, w=-0.002, h=0.008037272727272737, page=2
            ),
            mmda_box.Box(
                l=1.0, t=0.32670311318181816, w=0.0, h=0.010037272727272737, page=2
            ),
        ]
