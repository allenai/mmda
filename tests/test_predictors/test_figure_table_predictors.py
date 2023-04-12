import pickle
import unittest
from collections import defaultdict
import pathlib
import pytest

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions
from mmda.types import Document
from mmda.types.box import Box
from mmda.types.span import Span



class TestFigureCaptionPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent
        with open(cls.fixture_path / 'fixtures/doc_fixture_0c027af0ee9c1901c57f6579d903aedee7f4.pkl',
                  'rb') as file_handle:
            doc_json = pickle.load(file_handle)
            cls.doc = Document.from_json(doc_json)
        assert cls.doc.pages
        assert cls.doc.tokens
        assert cls.doc.blocks
        assert cls.doc.vila_span_groups
        cls.figure_table_predictor = FigureTablePredictions(cls.doc)

    def test_merge_boxes(self):
        result = self.figure_table_predictor.merge_boxes(self.doc.blocks, defaultdict(list))
        assert list(result[0].keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0][0], Span)

    def test_get_figure_caption_distance(self):
        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == 900

        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0), Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0))

        assert distance == pytest.approx(0.15)

    def test_predict(self):
        result = self.figure_table_predictor.predict()
        assert isinstance(result, tuple)
        assert [entry.type for entry in result[0]] == ['Figure', 'Figure', 'Figure', 'Figure']
        assert [entry.type for entry in result[1]] == ['Table', 'Table', 'Table', 'Table']
