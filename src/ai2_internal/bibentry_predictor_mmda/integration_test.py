"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import gzip
import json
import logging
import os
import pathlib
import sys
import unittest

from .. import api
from mmda.types.document import Document
from .interface import Instance, Prediction

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)


def resolve(file: str) -> str:
    return os.path.join(pathlib.Path(os.path.dirname(__file__)), "data", file)


def read_fixture_doc_and_entries(filename):
    path = resolve(filename)

    with gzip.open(path, "r") as f:
        raw = json.loads(f.read())

    doc = Document.from_json(raw["doc"])
    bib_entry_boxes = raw["bib_entry_boxes"]

    return doc, bib_entry_boxes


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__handles_no_bib_entries(self, container):
        filename = "test_data.json.gz"
        doc, bib_entry_boxes = read_fixture_doc_and_entries(filename)
        instance = Instance(
            symbols=doc.symbols,
            tokens=[api.SpanGroup.from_mmda(token) for token in doc.tokens],
            pages=[api.SpanGroup.from_mmda(page) for page in doc.pages],
            bib_entry_boxes=[]
        )
        prediction = container.predict_batch([instance])[0]

        self.assertEqual(prediction, Prediction(
            bib_entry_number=[],
            bib_entry_authors=[],
            bib_entry_title=[],
            bib_entry_venue_or_event=[],
            bib_entry_year=[],
            bib_entry_doi=[],
            bib_entry_url=[]
        ))

    def test__predictions(self, container):
        # Produced from running upstream models on example paper
        # (000026bab3c52aa8ff37dc3e155ffbcb506aa1f6.pdf)
        filename = "test_data.json.gz"
        doc, bib_entry_boxes = read_fixture_doc_and_entries(filename)
        instance = Instance(
            symbols=doc.symbols,
            tokens=[api.SpanGroup.from_mmda(token) for token in doc.tokens],
            pages=[api.SpanGroup.from_mmda(page) for page in doc.pages],
            bib_entry_boxes=[api.BoxGroup(**bib_entry) for bib_entry in bib_entry_boxes]
        )
        prediction = container.predict_batch([instance])[0]

        final_doc = instance.to_mmda()
        final_doc.annotate(bib_entry_number=[sg.to_mmda() for sg in prediction.bib_entry_number])
        final_doc.annotate(bib_entry_authors=[sg.to_mmda() for sg in prediction.bib_entry_authors])
        final_doc.annotate(bib_entry_title=[sg.to_mmda() for sg in prediction.bib_entry_title])
        final_doc.annotate(bib_entry_venue_or_event=[sg.to_mmda() for sg in prediction.bib_entry_venue_or_event])
        final_doc.annotate(bib_entry_doi=[sg.to_mmda() for sg in prediction.bib_entry_doi])
        final_doc.annotate(bib_entry_year=[sg.to_mmda() for sg in prediction.bib_entry_year])
        final_doc.annotate(bib_entry_url=[sg.to_mmda() for sg in prediction.bib_entry_url])

        # Make some assertions about a couple arbitrary entries in the bibliography

        # First entry is technically number "10" based on symbol start position only. Order came out of pdf oddly.
        self.assertEqual(final_doc.bib_entry_boxes[0].bib_entry_number[0].text, "10")
        self.assertEqual(final_doc.bib_entry_boxes[0].bib_entry_authors[0].text, "Srivastava, K.")
        self.assertEqual(final_doc.bib_entry_boxes[0].bib_entry_authors[1].text, "V.B. Upadhyay")
        self.assertEqual(final_doc.bib_entry_boxes[0].bib_entry_title[0].text, "Effect of Phytoecdysteroid on Length of Silk Filament and\nNon-Breakable Filament Length of Multivoltine\nMulberry Silkworm B. mori Linn")

        # *Actual* first entry
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_number[0].text, "1")
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_authors[0].text, "Upadhyay, V.B.")
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_authors[1].text, "K.P. Gaur")
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_authors[2].text, "S.K. Gupta")
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_title[0].text, "Effect of ecological factors on the silk producing\npotential of the mulberry silkworm ( Bombyx mori\nLinn. )")
        self.assertEqual(final_doc.bib_entry_boxes[1].bib_entry_venue_or_event[0].text, "Malays. Appl. Biol.")

    def test__empty_boxes_passes(self, container):
        # Produced from getting annotation store data from running upstream models on example paper
        # Upstream bib detector model produces annotations which result in SpanGroups with empty spans
        # See https://github.com/allenai/mmda/pull/186
        # (3cf45514384bbb7d083ae53e19bdc22300e648ab.pdf)
        filename = "test_data_v2_boxes_turn_into_empty_span_groups.json.gz"
        doc, bib_entry_boxes = read_fixture_doc_and_entries(filename)
        instance = Instance(
            symbols=doc.symbols,
            tokens=[api.SpanGroup.from_mmda(token) for token in doc.tokens],
            pages=[api.SpanGroup.from_mmda(page) for page in doc.pages],
            bib_entry_boxes=[api.BoxGroup(**bib_entry) for bib_entry in bib_entry_boxes]
        )
        prediction = container.predict_batch([instance])[0]

