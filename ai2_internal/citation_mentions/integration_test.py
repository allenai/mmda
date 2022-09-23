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
    {'start': 2161, 'end': 2162, 'id': 0, 'page': 0},
    {'start': 2165, 'end': 2166, 'id': 1, 'page': 0},
    {'start': 2303, 'end': 2304, 'id': 2, 'page': 0},
    {'start': 2307, 'end': 2308, 'id': 3, 'page': 0},
    {'start': 2548, 'end': 2550, 'id': 4, 'page': 0},
    {'start': 2553, 'end': 2555, 'id': 5, 'page': 0},
    {'start': 3100, 'end': 3102, 'id': 6, 'page': 0},
    {'start': 3105, 'end': 3107, 'id': 7, 'page': 0},
    {'start': 3130, 'end': 3132, 'id': 8, 'page': 0},
    {'start': 3165, 'end': 3167, 'id': 9, 'page': 0},
    {'start': 3170, 'end': 3172, 'id': 10, 'page': 0},
    {'start': 3420, 'end': 3422, 'id': 11, 'page': 0},
    {'start': 3425, 'end': 3427, 'id': 12, 'page': 0},
    {'start': 4875, 'end': 4877, 'id': 13, 'page': 1},
    {'start': 4986, 'end': 4988, 'id': 14, 'page': 1},
    {'start': 5229, 'end': 5231, 'id': 15, 'page': 1},
    {'start': 5234, 'end': 5236, 'id': 16, 'page': 1},
    {'start': 5239, 'end': 5241, 'id': 17, 'page': 1},
    {'start': 5244, 'end': 5246, 'id': 18, 'page': 1},
    {'start': 5372, 'end': 5374, 'id': 19, 'page': 1},
    {'start': 5377, 'end': 5379, 'id': 20, 'page': 1},
    {'start': 5617, 'end': 5619, 'id': 21, 'page': 1},
    {'start': 5622, 'end': 5624, 'id': 22, 'page': 1},
    {'start': 6068, 'end': 6070, 'id': 23, 'page': 1},
    {'start': 6073, 'end': 6075, 'id': 24, 'page': 1},
    {'start': 6308, 'end': 6310, 'id': 25, 'page': 1},
    {'start': 6313, 'end': 6315, 'id': 26, 'page': 1},
    {'start': 6739, 'end': 6741, 'id': 27, 'page': 1},
    {'start': 6744, 'end': 6746, 'id': 28, 'page': 1}
]


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        pdf_path = pathlib.Path(__file__).resolve().parent / "data" / "arxiv-1906.08632-pages1-2.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(str(pdf_path))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]

        instances = [Instance(symbols=doc.symbols, tokens=tokens, pages=pages)]
        predictions = container.predict_batch(instances)
        mentions = [
            {"start": span.start, "end": span.end, "id": mention.id, "page": span.box.page}
            for mention in predictions[0].mentions
            for span in mention.spans
        ]

        self.assertEqual(mentions, expected_mentions)
