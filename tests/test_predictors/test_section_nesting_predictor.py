"""
Tests for SectionNestingPredictor

@rauthur
"""

import pathlib
import unittest
from copy import deepcopy

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.xgb_predictors.section_nesting_predictor import (
    SectionNestingPredictor,
)
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import SpanGroup
from mmda.types.box import Box

S2VL_LABEL_MAP = {
    "0": "Title",
    "1": "Author",
    "2": "Abstract",
    "3": "Keywords",
    "4": "Section",
    "5": "Paragraph",
    "6": "List",
    "7": "Bibliography",
    "8": "Equation",
    "9": "Algorithm",
    "10": "Figure",
    "11": "Table",
    "12": "Caption",
    "13": "Header",
    "14": "Footer",
    "15": "Footnote",
}
S2VL_LABEL_MAP = {int(key): val for key, val in S2VL_LABEL_MAP.items()}


class TestSectionNestingPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.parser = PDFPlumberParser(extra_attrs=[])
        self.rasterizer = PDF2ImageRasterizer()
        self.layout_predictor = LayoutParserPredictor.from_pretrained(
            "lp://efficientdet/PubLayNet"
        )
        self.vila = IVILAPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
            agg_level="row",
            added_special_sepration_token="[BLK]",
        )

        self.predictor = SectionNestingPredictor(self.fixture_path / "nesting.bin")

    def test_finds_sections(self):
        input_pdf_path = (
            self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"
        )

        doc = self.parser.parse(input_pdf_path)
        images = self.rasterizer.rasterize(input_pdf_path, dpi=72)
        doc.annotate_images(images)

        blocks = self.layout_predictor.predict(doc)
        doc.annotate(blocks=blocks)

        results = self.vila.predict(doc)
        doc.annotate(results=results)

        # Extract sections from VILA predictions and re-add boxes
        vila_sections = []

        for i, span_group in enumerate(doc.results):
            if span_group.type != 4:
                continue

            # Boxes are not returned from VILA so reach into tokens
            token_spans = [deepcopy(t.spans) for t in span_group.tokens]
            token_spans = [span for l in token_spans for span in l]  # Flatten list

            # Maintain the text from VILA for each span group
            metadata = deepcopy(span_group.metadata)
            metadata.text = span_group.text

            vila_sections.append(
                SpanGroup(
                    spans=token_spans,
                    box_group=deepcopy(span_group.box_group),
                    id=i,  # Ensure some ID is created
                    doc=None,  # Allows calling doc.annotate(...)
                    metadata=metadata,
                )
            )

        doc.annotate(sections=vila_sections)

        nestings = self.predictor.predict(doc)
        self.assertEqual(18, len(nestings))

        for nesting in nestings:
            self.assertIsNotNone(nesting.metadata.parent_id)
