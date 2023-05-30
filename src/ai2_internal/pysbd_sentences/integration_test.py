import json
import logging
import os
import sys
import unittest

from .interface import Instance, Prediction

from ai2_internal import api
from mmda.types.document import Document


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


def get_test_instance() -> Instance:
    doc_file = resolve("test_doc.json")
    with open(doc_file) as f:
        doc = Document.from_json(json.load(f))

    tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

    return Instance(
        symbols=doc.symbols,
        tokens=tokens,
        pages=pages
    )


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instance = get_test_instance()
        instances = [instance]
        predictions = container.predict_batch(instances)

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertTrue(len(prediction.sentences) < len(instance.tokens))
