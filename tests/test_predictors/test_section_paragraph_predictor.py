"""



"""


import pathlib
import unittest

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.heuristic_predictors.section_paragraph_predictor import \
    SectionParagraphPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.token_classification_predictor import \
    IVILATokenClassificationPredictor


class TestSectionHeaderPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()
        self.lp_predictor = LayoutParserPredictor.from_pretrained('lp://efficientdet/PubLayNet')
        self.vila_predictor = IVILATokenClassificationPredictor.from_pretrained('allenai/ivila-block-layoutlm-finetuned-docbank')
        self.predictor = SectionParagraphPredictor()

        input_pdf_path = self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"
        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        images = self.rasterizer.rasterize(input_pdf_path=input_pdf_path, dpi=72)
        doc.annotate_images(images=images)
        lp_output = self.lp_predictor.predict(document=doc)
        doc.annotate(blocks=lp_output)
        vila_output = self.vila_predictor.predict(document=doc)
        doc.annotate(vila_span_groups=vila_output)

        self.doc = doc

    def test_finds_sections(self):

        results = self.predictor.predict(document=self.doc)
        

