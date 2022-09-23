"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

predict_batch(instances: List[Instance]) -> List[Prediction]
"""
import logging
import pathlib
import sys
import unittest

from ai2_internal import api
from ai2_internal.citation_mentions.interface import Instance
from mmda.parsers.pdfplumber_parser import PDFPlumberParser

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)


expected_mentions = [
    {"start": 2107, "end": 2110},
    {"start": 2245, "end": 2246},
    {"start": 2249, "end": 2250},
    {"start": 2489, "end": 2491},
    {"start": 2494, "end": 2496},
    {"start": 3035, "end": 3040},
    {"start": 3063, "end": 3065},
    {"start": 3098, "end": 3103},
    {"start": 3340, "end": 3345}
]


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        pdf_path = pathlib.Path(__file__).resolve().parent / "data" / "arxiv-1906.08632-page0.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(str(pdf_path))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        instances = [Instance(symbols=doc.symbols, tokens=tokens, pages=pages)]
        predictions = container.predict_batch(instances)
        mentions = [
            {"start": span.start, "end": span.end}
            for mention in predictions[0].mentions
            for span in mention.spans
        ]

        self.assertEqual(mentions, expected_mentions)
