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

import json
import logging
import os
import pathlib
import sys
import unittest

from .. import api
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import SpanGroup as MmdaSpanGroup
from mmda.types.image import tobase64
from .interface import Instance

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


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):

    def get_images(self, pdf_filename):
        rasterizer = PDF2ImageRasterizer()
        return rasterizer.rasterize(str(resolve(pdf_filename)), dpi=72)

    def test__predictions(self, container):
        pdf = "000026bab3c52aa8ff37dc3e155ffbcb506aa1f6.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(resolve(pdf))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        page_images = self.get_images(pdf)
        encoded_page_images = [tobase64(img) for img in page_images]

        doc.annotate_images(page_images)

        with open(resolve("vila_span_groups.json")) as f:
            vila_span_groups = [api.SpanGroup(**sg) for sg in json.load(f)["vila_span_groups"]]

        instances = [Instance(
            symbols=doc.symbols,
            tokens=tokens,
            rows=rows,
            pages=pages,
            page_images=encoded_page_images,
            vila_span_groups=vila_span_groups)]

        predictions = container.predict_batch(instances)

        for bib_entry in predictions[0].bib_entries:
            self.assertEqual(bib_entry.box_group.type, "bib_entry")

        for raw_box in predictions[0].raw_bib_entry_boxes:
            self.assertEqual(raw_box.type, "raw_model_prediction")

        expected_bib_count = 31
        self.assertEqual(len(predictions[0].bib_entries), expected_bib_count)
        self.assertEqual(len(predictions[0].raw_bib_entry_boxes), expected_bib_count)

    def test__no_resulting_bibs(self, container):
        pdf = "no_bibs.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(resolve(pdf))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        page_images = self.get_images(pdf)
        encoded_page_images = [tobase64(img) for img in page_images]

        doc.annotate_images(page_images)

        with open(resolve("no_bibs_vila_span_groups.json")) as f:
            vila_span_groups = [api.SpanGroup(**sg) for sg in json.load(f)["vila_span_groups"]]

        instances = [Instance(
            symbols=doc.symbols,
            tokens=tokens,
            rows=rows,
            pages=pages,
            page_images=encoded_page_images,
            vila_span_groups=vila_span_groups)]

        # should not error
        predictions = container.predict_batch(instances)

        self.assertEqual(len(predictions[0].bib_entries), 0)
        self.assertEqual(len(predictions[0].raw_bib_entry_boxes), 0)

    def test__spanless_bibs(self, container):
        pdf = "spanless_bibs_3cf45514384bbb7d083ae53e19bdc22300e648ab.pdf"  # 3cf45514384bbb7d083ae53e19bdc22300e648ab
        doc = PDFPlumberParser(split_at_punctuation=True).parse(resolve(pdf))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        page_images = self.get_images(pdf)
        encoded_page_images = [tobase64(img) for img in page_images]

        doc.annotate_images(page_images)

        with open(resolve("spanless_bibs_vila_span_groups.json")) as f:
            vila_span_groups = [api.SpanGroup.from_mmda(MmdaSpanGroup.from_json(sg)) for sg in json.load(f)["vila_span_groups"]]

        instances = [Instance(
            symbols=doc.symbols,
            tokens=tokens,
            rows=rows,
            pages=pages,
            page_images=encoded_page_images,
            vila_span_groups=vila_span_groups)]

        # should not error
        predictions = container.predict_batch(instances)

        # several bib boxes overlapped others, causing them to end up with no spans
        expected_bib_count = 99
        expected_raw_bib_count = 102

        self.assertEqual(len(predictions[0].bib_entries), expected_bib_count)
        self.assertEqual(len(predictions[0].raw_bib_entry_boxes), expected_raw_bib_count)
