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


expected_mentions_text = [
    "1",
    "2",
    "3",
    "9",
    "10",
    "12",
    "10",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "11",
    "19",
    "17",
    "18",
    "20",
    "22",
    "23",
    "28",
    "29",
    "32",
    "33",
    "38",
    "38",
    "39",
    "40",
    "44"
]


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        pdf_path = pathlib.Path(__file__).resolve().parent / "data" / "arxiv-1906.08632-pages1-2.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(str(pdf_path))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]

        instances = [Instance(symbols=doc.symbols, tokens=tokens, pages=pages, rows=rows)]
        predictions = container.predict_batch(instances)

        mmda_mentions = [mention.to_mmda() for mention in predictions[0].mentions]
        doc.annotate(mentions=mmda_mentions)

        mentions_text = [m.text for m in doc.mentions]

        self.assertEqual(mentions_text, expected_mentions_text)

    def test__multi_row_predictions(self, container):
        pdf_path = pathlib.Path(__file__).resolve().parent / "data" / "arxiv-2201.05673-page1.pdf"
        doc = PDFPlumberParser(split_at_punctuation=True).parse(str(pdf_path))

        tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
        pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
        rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]

        instances = [Instance(symbols=doc.symbols, tokens=tokens, pages=pages, rows=rows)]
        predictions = container.predict_batch(instances)

        mmda_mentions = [mention.to_mmda() for mention in predictions[0].mentions]
        doc.annotate(mentions=mmda_mentions)

        self.assertEqual(doc.mentions[0].text, "Silver and Veness 2010")
        self.assertEqual(doc.mentions[1].text, "Browne et al . 2012")
        self.assertEqual(doc.mentions[2].text, "Somani et al . 2013")
        self.assertEqual(len(doc.mentions[0].box_group.boxes), 1)
        self.assertEqual(len(doc.mentions[1].box_group.boxes), 2)
        self.assertEqual(len(doc.mentions[2].box_group.boxes), 1)
