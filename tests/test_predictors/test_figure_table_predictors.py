import unittest

import pytest

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions

import os
import pathlib

from mmda.types.box import Box
from mmda.types.span import Span

os.chdir(pathlib.Path(__file__).parent.parent)


class TestDictionaryWordPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.pdf_path = 'fixtures/0c027af0ee9c1901c57f6579d903aedee7f4.pdf'
        import os
        print(os.path.realpath(__file__))
        cls.figure_table_predictor = FigureTablePredictions(cls.pdf_path, dpi=72)

    def test_make_vision_predictions(self):
        self.figure_table_predictor.make_vision_prediction()
        assert 'vision_predictions' in self.figure_table_predictor.doc.fields

    def test_make_villa_predictions(self):
        result = self.figure_table_predictor.make_villa_predictions()
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert isinstance(list(result.values())[0][0], Span)

    def test_merge_boxes(self):
        self.figure_table_predictor.make_vision_prediction()
        result = self.figure_table_predictor.merge_boxes(self.figure_table_predictor.doc)
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0], Span)

    def test_get_figure_cation_distance(self):
        distance = FigureTablePredictions.get_figure_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == 900

        distance = FigureTablePredictions.get_figure_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == pytest.approx(0.15)

    def test_make_boxgroups(self):
        result = self.figure_table_predictor.make_boxgroups(0, Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))
        assert result == pytest.approx([122.4, 237.6, 183.6, 316.8])

    def test_map_caption_object(self):
        result = self.figure_table_predictor.map_caption_object()
        assert isinstance(result, dict)
        assert list(result.keys()) == [0, 2, 3, 7]
        assert [result[idx][0].to_json() for idx in list(result.keys())] == [
            '{"dpi": 72, "page": 0, "object_type": "Figure", "box": [307.34411621093756, '
            '283.10247802734375, 547.9175415039062, 386.5572204589844], "caption": '
            '"Figure 1 : Motivation of"}',
            '{"dpi": 72, "page": 2, "object_type": "Figure", "box": [53.39092254638671, '
            '75.6981201171875, 283.1786117553711, 242.29154968261722], "caption": "Figure '
            '2 : Given a feature map as an input , max pooling\\n( top right ) and the '
            'proposed G - pooling ( bottom right ) cre -\\nate different output '
            'downsampled feature map based on the\\ncharacteristics of spatial cluster . '
            'The feature map within\\nthe sliding window ( blue dot line ) indicates a '
            'spatial clus -\\nter . Max pooling takes the max value ignoring the '
            'spatial\\ncluster , while our G - pooling takes the interpolated value '
            'at\\nthe center location . ( White , gray and black represent three\\nvalues '
            'range from low to high . )"}',
            '{"dpi": 72, "page": 3, "object_type": "Figure", "box": [57.03529357910156, '
            '69.08863067626953, 282.17950439453125, 174.696044921875], "caption": "Figure '
            '3 : A FCN network architecture with G - pooling ."}',
            '{"dpi": 72, "page": 7, "object_type": "Figure", "box": [60.94582748413086, '
            '68.12191009521484, 552.6776695251465, 347.5070724487305], "caption": "Figure '
            '4 : Qualitative results of ISPRS Potsdam . White : road , blue : building , '
            'cyan : low vegetation , green : trees , yellow :\\ncars , red : clutter ."}']

