"""

@kylel

"""

import os
import unittest

from mmda.types import Document, SpanGroup, PILImage, BoxGroup
from mmda.recipes.recipe import CoreRecipe
from tests.test_recipes.core_recipe_fixtures import FIRST_1000_SYMBOLS, PAGE_JSON, \
    BASE64_PAGE_IMAGE, FIRST_10_VILA_JSONS, SEGMENT_OF_WORD_JSONS, FIRST_3_BLOCKS_JSON, \
    FIRST_5_ROWS_JSON, FIRST_10_TOKENS_JSON


def round_all_floats(d: dict):
    import numbers

    def formatfloat(x):
        return "%.4g" % float(x)

    def pformat(dictionary, function):
        if isinstance(dictionary, dict):
            return {key: pformat(value, function) for key, value in dictionary.items()}
        if isinstance(dictionary, list):
            return [pformat(element, function) for element in dictionary]
        if isinstance(dictionary, numbers.Number):
            return function(dictionary)
        return dictionary

    return pformat(d, formatfloat)


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.pdfpath = os.path.join(os.path.dirname(__file__), '../fixtures/1903.10676.pdf')
        self.recipe = CoreRecipe()
        self.doc = self.recipe.from_path(pdfpath=self.pdfpath)

    def test_correct_output(self):
        self.assertEqual(self.doc.symbols[:1000], FIRST_1000_SYMBOLS)
        self.assertDictEqual(self.doc.pages[0].to_json(), PAGE_JSON)
        self.assertEqual(self.doc.images[0].to_json(), BASE64_PAGE_IMAGE)
        self.assertListEqual([round_all_floats(t.to_json()) for t in self.doc.tokens[:10]],
                             round_all_floats(FIRST_10_TOKENS_JSON))
        self.assertListEqual([round_all_floats(r.to_json()) for r in self.doc.rows[:5]],
                             round_all_floats(FIRST_5_ROWS_JSON))
        self.assertListEqual([round_all_floats(b.to_json()) for b in self.doc.blocks[:3]],
                             round_all_floats(FIRST_3_BLOCKS_JSON))
        self.assertListEqual([round_all_floats(v.to_json()) for v in self.doc.vila_span_groups[:10]],
                             round_all_floats(FIRST_10_VILA_JSONS))
        self.assertListEqual([round_all_floats(w.to_json()) for w in self.doc.words[25:30]],
                             round_all_floats(SEGMENT_OF_WORD_JSONS))

    def test_to_from_json(self):
        doc_json = self.doc.to_json(with_images=True)
        doc2 = Document.from_json(doc_dict=doc_json)
        self.assertDictEqual(doc_json, doc2.to_json(with_images=True))

    def test_manual_create_using_annotate(self):
        """
        This tests whether one can manually reconstruct a Document without using from_json().
        Annotations on a Document are order-invariant once created, so you can see this since the
        fields are being annotated in a different order than they were computed.
        """
        doc_json = self.doc.to_json(with_images=True)

        doc2 = Document(symbols=doc_json['symbols'], metadata=doc_json['metadata'])
        assert doc2.symbols == doc_json['symbols'] == self.doc.symbols
        assert doc2.metadata.to_json() == doc_json['metadata'] == self.doc.metadata.to_json()

        images = [PILImage.frombase64(img) for img in doc_json['images']]
        doc2.annotate_images(images)
        assert doc2.images[0].to_json() == doc_json['images'][0] == self.doc.images[0].to_json()

        rows = [SpanGroup.from_json(span_group_dict=r) for r in doc_json['rows']]
        doc2.annotate(rows=rows)
        assert [r.to_json() for r in doc2.rows] == doc_json['rows'] == \
               [r.to_json() for r in self.doc.rows]

        vila_span_groups = [SpanGroup.from_json(span_group_dict=v)
                            for v in doc_json['vila_span_groups']]
        doc2.annotate(vila_span_groups=vila_span_groups)
        assert [v.to_json() for v in doc2.vila_span_groups] == doc_json['vila_span_groups'] == \
               [v.to_json() for v in self.doc.vila_span_groups]

        words = [SpanGroup.from_json(span_group_dict=w) for w in doc_json['words']]
        doc2.annotate(words=words)
        assert [w.to_json() for w in doc2.words] == doc_json['words'] == \
               [w.to_json() for w in self.doc.words]

        tokens = [SpanGroup.from_json(span_group_dict=t) for t in doc_json['tokens']]
        doc2.annotate(tokens=tokens)
        assert [t.to_json() for t in doc2.tokens] == doc_json['tokens'] == \
               [t.to_json() for t in self.doc.tokens]

        blocks = [SpanGroup.from_json(span_group_dict=b) for b in doc_json['blocks']]
        doc2.annotate(blocks=blocks)
        assert [b.to_json() for b in doc2.blocks] == doc_json['blocks'] == \
               [b.to_json() for b in self.doc.blocks]
