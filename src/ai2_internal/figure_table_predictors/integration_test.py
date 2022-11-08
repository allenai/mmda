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
import sys
import unittest
from pathlib import Path

from PIL import Image
from .. import api
from mmda.types.document import Document
from mmda.types.image import tobase64
from .interface import Instance
from ..api import SpanGroup

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)

def resolve(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'test_fixtures', file)

def get_test_instance() -> Instance:
    doc_file = resolve('test_doc_sha_d0450478c38dda61f9943f417ab9fcdb2ebeae0a.json')
    with open(doc_file) as f:
        doc = Document.from_json(json.load(f))

    tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
    rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
    vila_span_groups = [api.SpanGroup.from_mmda(sg) for sg in doc.vila_span_groups]
    layoutparser_span_groups = [api.SpanGroup.from_mmda(sg) for sg in doc.layoutparser_span_groups]

    return Instance(
        symbols=doc.symbols,
        images=[tobase64(image) for image in doc.images],
        tokens=tokens,
        rows=rows,
        pages=pages,
        vila_span_groups=vila_span_groups,
        layoutparser_span_groups=layoutparser_span_groups,
    )


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [get_test_instance()]
        predictions = container.predict_batch(instances)
        assert isinstance(predictions[0].figure_list[0], SpanGroup)
        assert isinstance(predictions[0].table_list[0], SpanGroup)
        assert [span_group.type for prediction in predictions for span_group in prediction.figure_list] == [
            'Figure', 'Figure', 'Figure', 'Figure', 'Figure']
        assert [span_group.type for prediction in predictions for span_group in prediction.table_list] == [
            'Table', 'Table', 'Table', 'Table', 'Table', 'Table',
        ]
