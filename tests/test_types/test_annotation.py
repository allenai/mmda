from mmda.types.annotation import BoxGroup
from mmda.types.box import Box
import unittest


class TestBoxGroup(unittest.TestCase):
    def setUp(cls) -> None:
        cls.box_group_json = {'boxes': [{'left': 0.1,
                                         'top': 0.6,
                                         'width': 0.36,
                                         'height': 0.221,
                                         'page': 0}],
                              'id': None,
                              'type': 'Text'}

    def test_from_json(self):
        self.assertIsInstance(BoxGroup.from_json(self.box_group_json), BoxGroup)
        self.assertEqual(BoxGroup.from_json(self.box_group_json).boxes,
                         [Box(l=0.1, t=0.6, w=0.36, h=0.221, page=0)])

        self.assertEqual(BoxGroup.from_json(self.box_group_json).id, None)
        self.assertEqual(BoxGroup.from_json(self.box_group_json).type, 'Text')

    def test_to_json(self):
        self.assertIsInstance(BoxGroup.from_json(self.box_group_json).to_json(), dict)
        self.assertEqual(BoxGroup.from_json(self.box_group_json).to_json()['boxes'],
                         [{'left': 0.1,
                           'top': 0.6,
                           'width': 0.36,
                           'height': 0.221,
                           'page': 0}])

        self.assertEqual(list(BoxGroup.from_json(self.box_group_json).to_json().keys()), ['boxes', 'metadata', 'uuid'])
