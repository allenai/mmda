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
import sys
import unittest

from .interface import Instance, Prediction
from ai2_internal import api
from mmda.types.document import Document


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fixtures")


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)


def resolve(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), "test_fixtures", file)


def read_fixture_doc(filename):
    path = resolve(filename)

    with gzip.open(path, "r") as f:
        doc_json = json.loads(f.read())

    doc = Document.from_json(doc_json)

    return doc


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        doc_file = resolve("test_doc.json")
        with open(doc_file) as f:
            doc = Document.from_json(json.load(f))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        instances = [Instance(symbols=doc.symbols, tokens=tokens)]
        predictions = container.predict_batch(instances)
        prediction = predictions[0]

        self.assertTrue(len(prediction.words) <= len(tokens))
