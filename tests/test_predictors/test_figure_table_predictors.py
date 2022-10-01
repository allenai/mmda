import unittest

import pytest

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions

import os
import pathlib

from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.span import Span

os.chdir(pathlib.Path(__file__).parent.parent)


class TestDictionaryWordPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.pdf_path = 'fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf'
        cls.figure_table_predictor = FigureTablePredictions(dpi=72)
        cls.figure_table_predictor.create_doc_rasterize(cls.pdf_path)

    def test_make_vision_predictions(self):
        self.figure_table_predictor.make_vision_prediction(self.figure_table_predictor.doc)
        assert 'layoutparser_span_groups' in self.figure_table_predictor.doc.fields

    def test_make_villa_predictions(self):
        result = self.figure_table_predictor.make_villa_predictions(self.figure_table_predictor.doc)
        assert isinstance(result, Document)
        assert 'vila_span_groups' in result.fields

    def test_merge_boxes(self):
        doc = self.figure_table_predictor.make_vision_prediction(self.figure_table_predictor.doc)
        result = self.figure_table_predictor.merge_boxes(doc.layoutparser_span_groups)
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0], Span)

    def test_get_figure_cation_distance(self):
        distance = FigureTablePredictions.get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == 900

        distance = FigureTablePredictions.get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == pytest.approx(0.15)

    def test_make_boxgroups(self):
        result = self.figure_table_predictor.make_boxgroups(self.figure_table_predictor.doc,
                                                            0, Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))
        assert result == pytest.approx([122.4, 237.6, 183.6, 316.8])

    def test_predict(self):
        FigureTablePredictions.make_vision_prediction(self.figure_table_predictor.doc)
        FigureTablePredictions.make_villa_predictions(self.figure_table_predictor.doc)
        result = self.figure_table_predictor.predict(self.figure_table_predictor.doc)
        assert isinstance(result, list)
        assert [{'box_group': {'boxes': [[0.5021962683185254,
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
                   'text': 'Figure 2 : Given a feature map as an input , max '
                           'pooling\n'
                           '( top right ) and the proposed G - pooling ( bottom '
                           'right ) cre -\n'
                           'ate different output downsampled feature map based on '
                           'the\n'
                           'characteristics of spatial cluster . The feature map '
                           'within\n'
                           'the sliding window ( blue dot line ) indicates a '
                           'spatial clus -\n'
                           'ter . Max pooling takes the max value ignoring the '
                           'spatial\n'
                           'cluster , while our G - pooling takes the interpolated '
                           'value at\n'
                           'the center location . ( White , gray and black '
                           'represent three\n'
                           'values range from low to high . )',
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
                   'text': 'Figure 3 : A FCN network architecture with G - pooling '
                           '.',
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
                   'text': 'Figure 4 : Qualitative results of ISPRS Potsdam . '
                           'White : road , blue : building , cyan : low vegetation '
                           ', green : trees , yellow :\n'
                           'cars , red : clutter .',
                   'type': 'Figure'},
      'spans': [{'box': [0.08188235294117648,
                         0.4603690929292929,
                         0.8088263784313716,
                         0.02767373737373735,
                         7],
                 'end': 33031,
                 'start': 32877}],
      'uuid': 'bd67442f-f8e5-4b76-9d21-c51fab68662a'}]

