"""

Test extraction of outline metadata from a PDF.

@rauthur

"""

import pathlib
import unittest

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.queriers.pdfminer_outline_querier import (
    OutlineMetadata,
    PDFMinerOutlineParserError,
    PDFMinerOutlineQuerier,
)


class TestPDFMinerOutlineQuerier(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.parser = PDFPlumberParser()
        self.querier = PDFMinerOutlineQuerier()

    def test_query(self):
        input_pdf_path = (
            self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"
        )

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        self.querier.query(input_pdf_path=input_pdf_path, doc=doc)

        self.assertIsNotNone(doc.metadata.outlines)
        self.assertEqual(18, len(doc.metadata.outlines))

        x = OutlineMetadata.from_metadata_dict(doc.metadata.outlines[0])
        self.assertEqual("I Introduction", x.title)
        self.assertEqual(0, x.level)

        x = OutlineMetadata.from_metadata_dict(doc.metadata.outlines[4])
        self.assertEqual("IV-A Overview", x.title)
        self.assertEqual(1, x.level)

    def test_raise_exceptions(self):
        input_pdf_path = self.fixture_path / "1903.10676.pdf"
        doc = self.parser.parse(input_pdf_path=input_pdf_path)

        with self.assertRaises(PDFMinerOutlineParserError):
            self.querier.query(
                input_pdf_path=input_pdf_path, doc=doc, raise_exceptions=True
            )

    def test_swallow_exceptions(self):
        input_pdf_path = self.fixture_path / "1903.10676.pdf"

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        self.querier.query(input_pdf_path=input_pdf_path, doc=doc)

        self.assertEqual(0, len(doc.metadata.outlines))

    def test_does_not_capture_file_missing_exception(self):
        input_pdf_path = self.fixture_path / "this-pdf-does-not-exist.pdf"
        doc = None

        with self.assertRaises(FileNotFoundError):
            self.querier.query(input_pdf_path=input_pdf_path, doc=doc)
