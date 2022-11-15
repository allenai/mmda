"""
Tests for SectionHeaderPredictor

@rauthur
"""

import pathlib
import unittest

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.heuristic_predictors.section_header_predictor import (
    SectionHeaderPredictor,
)
from mmda.queriers.pdfminer_outline_querier import PDFMinerOutlineQuerier


class TestSectionHeaderPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.parser = PDFPlumberParser()
        self.querier = PDFMinerOutlineQuerier()
        self.predictor = SectionHeaderPredictor()

    def test_finds_sections(self):
        input_pdf_path = (
            self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"
        )

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        self.querier.query(input_pdf_path=input_pdf_path, doc=doc)
        doc.annotate(sections=self.predictor.predict(document=doc))

        self.assertEqual(18, len(doc.sections))
