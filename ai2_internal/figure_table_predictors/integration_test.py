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
"""
import json
import logging
import sys
import unittest

from mmda.types import Document
from .interface import Instance

from PIL import Image
from pathlib import Path

import os.path

from mmda.types.image import tobase64
from .. import api

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

def resolve_image(file: str) -> str:
    return os.path.join(Path(os.path.dirname(__file__)).parent, "shared_test_fixtures", file)

def get_test_instance() -> Instance:
    doc_file = resolve("test_doc.json")
    with open(doc_file) as f:
        doc = Document.from_json(json.load(f))

    tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
    rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

    with open(resolve("test_blocks.json")) as f:
        blocks = [api.BoxGroup(**bg) for bg in json.load(f)["groups"]]

    images = [tobase64(Image.open(resolve_image("page0.png")))]

    return Instance(
        symbols=doc.symbols,
        images=images,
        tokens=tokens,
        rows=rows,
        pages=pages,
        blocks=blocks,
    )

@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):

    def get_images(self):
        return [
            tobase64(
                Image.open(os.path.join(Path(os.path.dirname(__file__)).parent, "shared_test_fixtures", "page0.png"))
            )
        ]

    def test__predictions(self, container):

        instances = [Instance(page_images=self.get_images())]
        predictions = container.predict_batch(instances)

        for prediction in predictions:
            self.assertEqual(
                set(g.type for g in prediction.groups),
                {"Text", "Title"},
            )
