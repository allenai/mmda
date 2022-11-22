"""

Test extraction of outline metadata from a PDF.

@rauthur

"""

import pathlib
import unittest

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.utils.outline_metadata import (
    Outline,
    PDFMinerOutlineExtractor,
    PDFMinerOutlineExtractorError,
)


class TestPDFMinerOutlineExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.parser = PDFPlumberParser()
        self.extractor = PDFMinerOutlineExtractor()

    def test_query(self):
        input_pdf_path = (
            self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"
        )

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        self.extractor.extract(input_pdf_path=input_pdf_path, doc=doc)

        self.assertIsNotNone(doc.metadata.outline)
        self.assertEqual(18, len(doc.metadata.outline["items"]))

        outline = Outline.from_metadata_dict(doc.metadata)

        x = outline.items[0]
        self.assertEqual("I Introduction", x.title)
        self.assertEqual(0, x.level)

        x = outline.items[4]
        self.assertEqual("IV-A Overview", x.title)
        self.assertEqual(1, x.level)

    def test_raise_exceptions(self):
        input_pdf_path = self.fixture_path / "1903.10676.pdf"
        doc = self.parser.parse(input_pdf_path=input_pdf_path)

        with self.assertRaises(PDFMinerOutlineExtractorError):
            self.extractor.extract(
                input_pdf_path=input_pdf_path, doc=doc, raise_exceptions=True
            )

    def test_swallow_exceptions(self):
        input_pdf_path = self.fixture_path / "1903.10676.pdf"

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        self.extractor.extract(input_pdf_path=input_pdf_path, doc=doc)

        self.assertEqual(0, len(doc.metadata.outline["items"]))

    def test_does_not_capture_file_missing_exception(self):
        input_pdf_path = self.fixture_path / "this-pdf-does-not-exist.pdf"
        doc = None

        with self.assertRaises(FileNotFoundError):
            self.extractor.extract(input_pdf_path=input_pdf_path, doc=doc)
