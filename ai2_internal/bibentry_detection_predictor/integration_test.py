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

import logging
import os
import pathlib
import sys
import unittest

from .interface import Instance

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.token_classification_predictor import IVILATokenClassificationPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types import api
from mmda.types.image import tobase64

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)

pdf_path = os.path.join(pathlib.Path(os.path.dirname(__file__)).parent, "shared_test_fixtures",
                        "26bab3c52aa8ff37dc3e155ffbcb506aa1f6.pdf")

@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):

    def get_images(self):
        rasterizer = PDF2ImageRasterizer()
        return rasterizer.rasterize(str(pdf_path), dpi=72)

    def test__predictions(self, container):
        doc = PDFPlumberParser(split_at_punctuation=True).parse(pdf_path)

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        page_images = self.get_images()
        encoded_page_images = [tobase64(img) for img in page_images]

        doc.annotate_images(page_images)

        ivilaA = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
        )

        vila_span_groups = [api.SpanGroup.from_mmda(sg) for sg in ivilaA.predict(doc, subpage_per_run=2)]

        instances = [Instance(
            symbols=doc.symbols,
            tokens=tokens,
            rows=rows,
            pages=pages,
            page_images=encoded_page_images,
            vila_span_groups=vila_span_groups)]

        predictions = container.predict_batch(instances)

        for bib_entry in predictions[0].bib_entry_boxes:
            self.assertEqual(bib_entry.type, "bib_entry")
            self.assertEqual(bib_entry.box_group.type, "processed_box")

        for original_box in predictions[0].raw_bib_entry_boxes:
            self.assertEqual(original_box.type, "raw_model_prediction")
