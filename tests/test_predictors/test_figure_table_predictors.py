import json
import pathlib
import pickle
import unittest
from collections import defaultdict

import pytest

from ai2_internal.api import Relation
from mmda.predictors.heuristic_predictors.figure_table_predictors import (
    FigureTablePredictions,
)
from mmda.types import Box, BoxGroup, Document, Span


class TestFigureCaptionPredictor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent
        with open(
            cls.fixture_path
            / "fixtures/doc_fixture_e5910c027af0ee9c1901c57f6579d903aedee7f4.pkl",
            "rb",
        ) as file_handle:
            doc_json = pickle.load(file_handle)
            cls.doc = Document.from_json(doc_json)
        assert cls.doc.pages
        assert cls.doc.tokens
        assert cls.doc.blocks
        assert cls.doc.vila_span_groups
        with open(
            cls.fixture_path
            / "fixtures/doc_fixture_2149e0c1106e6dfa36ea787167d6611cf88b69cb.json",
            "rb",
        ) as file_handle:
            dic_json = json.load(file_handle)
            cls.doc_2 = Document.from_json(dic_json["doc"])
            layout_equations = [
                BoxGroup.from_json(entry) for entry in dic_json["layout_equations"]
            ]
            cls.doc_2.annotate(blocks=layout_equations)

        with open(
            cls.fixture_path / "fixtures/figure_table_predictions.json", "r"
        ) as file:
            cls.figure_predictions = json.load(file)

        cls.figure_table_predictor = FigureTablePredictions(cls.doc)

    def test_merge_boxes(self):
        result = self.figure_table_predictor.merge_boxes(
            self.doc.blocks, defaultdict(list)
        )
        assert list(result[0].keys()) == [0, 2, 3, 7]
        assert isinstance(result[0][0][0], Span)

    def test_get_figure_caption_distance(self):
        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0),
            Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0),
        )

        assert distance == 900

        distance = FigureTablePredictions._get_object_caption_distance(
            Box(l=0.2, t=0.2, w=0.1, h=0.1, page=0),
            Box(l=0.2, t=0.3, w=0.1, h=0.1, page=0),
        )

        assert distance == pytest.approx(0.15)

    def test_generate_map_of_layout_to_tokens(self):
        """
        Test that the function generates a map of layout to tokens using
        """
        vila_caption = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups,
            caption_content="fig",
            span_group_types=["Caption"],
        )

        vila_caption_dict = FigureTablePredictions._create_dict_of_pages_spans_vila(
            vila_caption
        )
        result = self.figure_table_predictor.generate_map_of_layout_to_tokens(
            vila_caption_dict, defaultdict(list), defaultdict(list)
        )
        assert list(result.keys()) == []

    def test_predict_e5910c027af0ee9c1901c57f6579d903aedee7f4(self):
        """
        Test that the function generates a map of layout to tokens using
        for doc_fixture_e5910c027af0ee9c1901c57f6579d903aedee7f4.pkl
        """
        result = self.figure_table_predictor.predict()
        assert isinstance(result, dict)
        assert list(result.keys()) == [
            "figures",
            "figure_captions",
            "figure_to_figure_captions",
            "tables",
            "table_captions",
            "table_to_table_captions",
        ]
        assert len(result["figures"]) == 4
        assert len(result["tables"]) == 4
        assert isinstance(result["figure_to_figure_captions"][0], Relation)
        assert isinstance(result["table_to_table_captions"][0], Relation)
        assert [figure.to_json() for figure in result["figures"]] == [
            {
                "boxes": [
                    {
                        "height": 0.130624674787425,
                        "left": 0.5021962683185254,
                        "page": 0,
                        "top": 0.3574526237718987,
                        "width": 0.3930938321780535,
                    }
                ]
            },
            {
                "boxes": [
                    {
                        "height": 0.21034525861643782,
                        "left": 0.08724006952023973,
                        "page": 2,
                        "top": 0.09557842485832446,
                        "width": 0.3754700804068372,
                    }
                ],
                "id": 1,
            },
            {
                "boxes": [
                    {
                        "height": 0.31222110318652835,
                        "left": 0.08188235294117646,
                        "page": 3,
                        "top": 0.08723311954074436,
                        "width": 0.37919526861851516,
                    }
                ],
                "id": 2,
            },
            {
                "boxes": [
                    {
                        "height": 0.3527590433756511,
                        "left": 0.09958468543158637,
                        "page": 7,
                        "top": 0.08601251274648339,
                        "width": 0.8034834020278033,
                    }
                ],
                "id": 3,
            },
        ]
        assert [
            figure_caption.to_json() for figure_caption in result["figure_captions"]
        ] == [
            {"id": 0, "metadata": {}, "spans": [{"end": 2057, "start": 2034}]},
            {"id": 1, "metadata": {}, "spans": [{"end": 9679, "start": 9175}]},
            {"id": 2, "metadata": {}, "spans": [{"end": 13875, "start": 13822}]},
            {"id": 3, "metadata": {}, "spans": [{"end": 31364, "start": 31224}]},
        ]

        assert [table.to_json() for table in result["tables"]] == [
            {
                "boxes": [
                    {
                        "height": 0.2796805025351168,
                        "left": 0.16789371515411178,
                        "page": 4,
                        "top": 0.1370883614125878,
                        "width": 0.6443845462175756,
                    }
                ]
            },
            {
                "boxes": [
                    {
                        "height": 0.20913203075678666,
                        "left": 0.1747694701151131,
                        "page": 5,
                        "top": 0.13721680882001164,
                        "width": 0.622537251391442,
                    }
                ],
                "id": 1,
            },
            {
                "boxes": [
                    {
                        "height": 0.06003320096719145,
                        "left": 0.15402431114047183,
                        "page": 5,
                        "top": 0.5840287642045454,
                        "width": 0.2569979998021344,
                    }
                ],
                "id": 2,
            },
            {
                "boxes": [
                    {
                        "height": 0.23519277090978136,
                        "left": 0.5027104296715431,
                        "page": 6,
                        "top": 0.27805763784081045,
                        "width": 0.3950077131682751,
                    }
                ],
                "id": 3,
            },
        ]

        assert [
            table_caption.to_json() for table_caption in result["table_captions"]
        ] == [
            {"id": 0, "metadata": {}, "spans": [{"end": 18359, "start": 18198}]},
            {"id": 1, "metadata": {}, "spans": [{"end": 22214, "start": 22042}]},
            {"id": 2, "metadata": {}, "spans": [{"end": 23502, "start": 23400}]},
            {"id": 3, "metadata": {}, "spans": [{"end": 29584, "start": 29369}]},
        ]

    def test_predict_2149e0c1106e6dfa36ea787167d6611cf88b69cb(self):
        """
        Test that the function generates a map of layout to tokens using
        for doc_fixture_2149e0c1106e6dfa36ea787167d6611cf88b69cb.json
        """
        self.figure_table_predictor.doc = self.doc_2
        result = self.figure_table_predictor.predict()
        assert isinstance(result, dict)
        assert list(result.keys()) == [
            "figures",
            "figure_captions",
            "figure_to_figure_captions",
            "tables",
            "table_captions",
            "table_to_table_captions",
        ]
        assert len(result["figures"]) == 19
        assert len(result["tables"]) == 0
        assert isinstance(result["figure_to_figure_captions"][0], Relation)
        assert [
            figure.to_json() for figure in result["figures"]
        ] == self.figure_predictions
        assert [
            figure_caption.to_json() for figure_caption in result["figure_captions"]
        ] == [
            {"id": 0, "metadata": {}, "spans": [{"end": 5253, "start": 5019}]},
            {"id": 1, "metadata": {}, "spans": [{"end": 9230, "start": 8976}]},
            {"id": 2, "metadata": {}, "spans": [{"end": 13164, "start": 12935}]},
            {"id": 3, "metadata": {}, "spans": [{"end": 17600, "start": 17373}]},
            {"id": 4, "metadata": {}, "spans": [{"end": 23624, "start": 23205}]},
            {"id": 5, "metadata": {}, "spans": [{"end": 21009, "start": 20070}]},
            {"id": 6, "metadata": {}, "spans": [{"end": 28975, "start": 28838}]},
            {"id": 7, "metadata": {}, "spans": [{"end": 32839, "start": 32681}]},
            {"id": 8, "metadata": {}, "spans": [{"end": 37061, "start": 36394}]},
            {"id": 9, "metadata": {}, "spans": [{"end": 42245, "start": 42063}]},
            {"id": 10, "metadata": {}, "spans": [{"end": 43512, "start": 43418}]},
            {"id": 11, "metadata": {}, "spans": [{"end": 46726, "start": 46542}]},
            {"id": 12, "metadata": {}, "spans": [{"end": 50359, "start": 50192}]},
            {"id": 13, "metadata": {}, "spans": [{"end": 57779, "start": 57323}]},
            {"id": 14, "metadata": {}, "spans": [{"end": 60918, "start": 60838}]},
            {"id": 15, "metadata": {}, "spans": [{"end": 64943, "start": 64238}]},
            {"id": 16, "metadata": {}, "spans": [{"end": 69170, "start": 68548}]},
            {"id": 17, "metadata": {}, "spans": [{"end": 75951, "start": 75767}]},
            {"id": 18, "metadata": {}, "spans": [{"end": 80129, "start": 79561}]},
        ]
