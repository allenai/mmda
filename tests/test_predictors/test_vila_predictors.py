import json
import json
import pathlib
import unittest

from PIL import Image

from mmda.types.document import Document
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor, HVILAPredictor
from mmda.predictors.hf_predictors.token_classification_predictor import (
    IVILATokenClassificationPredictor,
    HVILATokenClassificationPredictor,
)


class TestFigureVilaPredictors(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent / 'fixtures'

        cls.DOCBANK_LABEL_MAP = {
            "0": "paragraph",
            "1": "title",
            "2": "equation",
            "3": "reference",
            "4": "section",
            "5": "list",
            "6": "table",
            "7": "caption",
            "8": "author",
            "9": "abstract",
            "10": "footer",
            "11": "date",
            "12": "figure",
        }
        cls.DOCBANK_LABEL_MAP = {int(key): val for key, val in cls.DOCBANK_LABEL_MAP.items()}

        cls.S2VL_LABEL_MAP = {
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

        cls.S2VL_LABEL_MAP = {int(key): val for key, val in cls.S2VL_LABEL_MAP.items()}

    def test_vila_predictors(self):
        layout_predictor = LayoutParserPredictor.from_pretrained(
            "lp://efficientdet/PubLayNet"
        )

        pdfplumber_parser = PDFPlumberParser()
        rasterizer = PDF2ImageRasterizer()

        doc = pdfplumber_parser.parse(input_pdf_path=self.fixture_path / "1903.10676.pdf")
        images = rasterizer.rasterize(input_pdf_path=self.fixture_path / "1903.10676.pdf", dpi=72)
        doc.annotate_images(images)

        layout_regions = layout_predictor.predict(doc)
        doc.annotate(blocks=layout_regions)

        ivilaA = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-docbank"
        )
        resA = ivilaA.predict(doc, subpage_per_run=2)
        del ivilaA

        ivilaB = IVILAPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-docbank",
            agg_level="block",
            added_special_sepration_token="[BLK]",
        )

        resB = ivilaB.predict(doc)
        del ivilaB

        assert [ele.spans for ele in resA] == [ele.spans for ele in resB]
        assert [ele.type for ele in resA] == [self.DOCBANK_LABEL_MAP[ele.type] for ele in resB]

        hvilaA = HVILATokenClassificationPredictor.from_pretrained(
            "allenai/hvila-row-layoutlm-finetuned-docbank"
        )
        resA = hvilaA.predict(doc, subpage_per_run=2)
        del hvilaA

        hvilaB = HVILAPredictor.from_pretrained(
            "allenai/hvila-row-layoutlm-finetuned-docbank",
            agg_level="row",
            added_special_sepration_token="[BLK]",
        )
        resB = hvilaB.predict(doc)
        del hvilaB

        assert [ele.spans for ele in resA] == [ele.spans for ele in resB]
        assert [ele.type for ele in resA] == [self.DOCBANK_LABEL_MAP[ele.type] for ele in resB]


        ivilaA = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
        )
        resA = ivilaA.predict(doc, subpage_per_run=2)
        del ivilaA

        ivilaB = IVILAPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
            agg_level="row",
            added_special_sepration_token="[BLK]",
        )

        resB = ivilaB.predict(doc)
        del ivilaB

        assert [ele.spans for ele in resA] == [ele.spans for ele in resB]
        assert [ele.type for ele in resA] == [self.S2VL_LABEL_MAP[ele.type] for ele in resB]

    def test_vila_predictors_with_special_unicode_inputs(self):

        test_doc_path = self.fixture_path / "unicode-test.json"

        with open(test_doc_path, 'r') as fp:
            res = json.load(fp)

        doc = Document.from_json(res)
        doc.annotate_images([Image.new("RGB", (596, 842))])

        ivilaA = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
        )

        ivilaA.predict(doc, subpage_per_run=2)