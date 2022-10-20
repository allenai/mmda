import unittest
from collections import defaultdict

import pytest

from mmda.predictors import LayoutParserPredictor
from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions

import os
import pathlib

from mmda.predictors.hf_predictors.token_classification_predictor import IVILATokenClassificationPredictor
from mmda.rasterizers import PDF2ImageRasterizer
from mmda.types.box import Box
from mmda.types.span import Span
from mmda.parsers.pdfplumber_parser import PDFPlumberParser



class TestFigureCaptionPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent
        cls.doc = (PDFPlumberParser()
                   .parse(input_pdf_path=os.path.join(cls.fixture_path,
                                                      'fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf')))
        cls.images = PDF2ImageRasterizer().rasterize(input_pdf_path=os.path.join(
            cls.fixture_path, 'fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf'),
                                                     dpi=72)
        assert cls.doc.pages
        assert cls.doc.tokens
        cls.doc.annotate_images(images=cls.images)
        vision_predictor = LayoutParserPredictor.from_pretrained()
        layoutparser_span_groups = vision_predictor.predict(document=cls.doc)
        cls.doc.annotate(layoutparser_span_groups=layoutparser_span_groups)
        vila_predictor = IVILATokenClassificationPredictor.from_pretrained(
            'allenai/ivila-row-layoutlm-finetuned-s2vl-v2')
        vila_span_groups = vila_predictor.predict(document=cls.doc)
        cls.doc.annotate(vila_span_groups=vila_span_groups)

    def test_merge_boxes(self):
        result = FigureTablePredictions._merge_boxes(self.doc.layoutparser_span_groups, defaultdict(list))
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0], Span)

    def test_get_figure_caption_distance(self):
        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == 900

        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == pytest.approx(0.15)

    def test_predict(self):
        result = FigureTablePredictions.predict(self.doc)
        assert isinstance(result, tuple)
        assert [entry.type for entry in result[0]] == ['Figure', 'Figure', 'Figure', 'Figure']
        assert [entry.type for entry in result[1]] == ['Table', 'Table', 'Table', 'Table']
