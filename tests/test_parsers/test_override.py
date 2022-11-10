import os
import pathlib
import unittest
from typing import List

from mmda.types.document import Document
from mmda.types.annotation import SpanGroup
from mmda.types.names import TokensField
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.base_predictors.base_predictor import BasePredictor


PDF_FIXTURE = (
    pathlib.Path(__file__).parent.parent
    / "fixtures/1903.10676.pdf"
)


class MockPredictor(BasePredictor):
    REQUIRED_BACKENDS = []  # pyright: ignore
    REQUIRED_DOCUMENT_FIELDS = []   # pyright: ignore

    def predict(self, document: Document) -> List[SpanGroup]:
        token: SpanGroup
        return [
            SpanGroup(
                spans=token.spans,
                box_group=token.box_group,
                metadata=token.metadata,
            )
            for token in getattr(document, TokensField, [])
        ]


class TestPDFPlumberParser(unittest.TestCase):

    def test_parse(self):
        parser = PDFPlumberParser()
        mock_predictor = MockPredictor()

        doc = parser.parse(input_pdf_path=str(PDF_FIXTURE))

        tokens = mock_predictor.predict(doc)

        # this should fail because we haven't specified an override
        with self.assertRaises(AssertionError):
            doc.annotate(tokens=tokens)

        doc.annotate(tokens=tokens, is_overwrite=True)
