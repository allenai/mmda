import unittest

from mmda.types.document import Document


class TestDocument(unittest.TestCase):
    def test__empty_annotations_work(self):
        doc = Document("This is a test document!")
        annotations = []
        doc.annotate(my_cool_field=annotations)
        self.assertEqual(doc.my_cool_field, [])

