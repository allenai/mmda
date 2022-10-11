import unittest

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

os.chdir(pathlib.Path(__file__).parent.parent)


class TestDictionaryWordPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.doc = (PDFPlumberParser()
                   .parse(input_pdf_path='fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf'))
        cls.images = PDF2ImageRasterizer().rasterize(input_pdf_path='fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf',
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
        result = FigureTablePredictions._merge_boxes(self.doc.layoutparser_span_groups)
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
        assert ([{'box_group': {'boxes': [[0.5021962683185254,
                                           0.3574526237718987,
                                           0.3930938321780535,
                                           0.130624674787425,
                                           0]],
                                'metadata': {'id': None, 'type': 'Figure'},
                                'uuid': 'd6cb60fd-4434-44b9-af59-ef4af7dacc07'},
                  'metadata': {'id': None,
                               'text': 'Figure 1 : Motivation of',
                               'type': 'Figure'},
                  'spans': [{'box': [0.5046764705882353,
                                     0.505044597979798,
                                     0.15726908267973871,
                                     0.012579040404040343,
                                     0],
                             'end': 2127,
                             'start': 2103}],
                  'uuid': '48c6f878-7d8d-40f9-aab8-b301904fd3ff'},
                 {'box_group': {'boxes': [[0.08724006952023973,
                                           0.09557843449139836,
                                           0.3754700804068372,
                                           0.21034523935029004,
                                           2]],
                                'metadata': {'id': None, 'type': 'Figure'},
                                'uuid': 'f7ed0df8-1fc0-41ac-983c-8f482d9c0aa4'},
                  'metadata': {'id': None,
                               'text': None,
                               'type': 'Figure'},
                  'spans': [{'box': [0.08188235294117646,
                                     0.3222478808080807,
                                     0.3860376222222223,
                                     0.13333787878787873,
                                     2],
                             'end': 10043,
                             'start': 9519}],
                  'uuid': '8d3e1417-b118-4f89-a860-55b225edf185'},
                 {'box_group': {'boxes': [[0.09319492414885876,
                                           0.08723311954074436,
                                           0.36788269741083285,
                                           0.13334269475455235,
                                           3]],
                                'metadata': {'id': None, 'type': 'Figure'},
                                'uuid': '18081ee9-1458-4dc4-a9a1-d83cbe643313'},
                  'metadata': {'id': None,
                               'text': None,
                               'type': 'Figure'},
                  'spans': [{'box': [0.09644934640522876,
                                     0.23085498030303023,
                                     0.3568952637254902,
                                     0.012855779292929359,
                                     3],
                             'end': 14366,
                             'start': 14310}],
                  'uuid': '653d1729-e99e-432e-8612-35b6bf47f96d'},
                 {'box_group': {'boxes': [[0.09958468543158637,
                                           0.08601251274648339,
                                           0.8034834020278033,
                                           0.3527590433756511,
                                           7]],
                                'metadata': {'id': None, 'type': 'Figure'},
                                'uuid': 'edd04791-a07f-4991-bf1e-6737a9f23db3'},
                  'metadata': {'id': None,
                               'text': None,
                               'type': 'Figure'},
                  'spans': [{'box': [0.08188235294117648,
                                     0.4603690929292929,
                                     0.8088263784313716,
                                     0.02767373737373735,
                                     7],
                             'end': 33031,
                             'start': 32877}],
                  'uuid': 'bd67442f-f8e5-4b76-9d21-c51fab68662a'}], [])
