import json
import pickle
import unittest
from collections import defaultdict
import pathlib
import pytest

from ai2_internal.api import Relation
from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions
from mmda.types import Document, BoxGroup
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
        with open(cls.fixture_path / 'fixtures/doc_fixture_2149e0c1106e6dfa36ea787167d6611cf88b69cb.json',
                  'rb') as file_handle:
            dic_json = json.load(file_handle)
            cls.doc_2 = Document.from_json(dic_json['doc'])
            layout_equations = [BoxGroup.from_json(entry) for entry in dic_json['layout_equations']]
            cls.doc_2.annotate(blocks=layout_equations)

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

    def test_generate_map_of_layout_to_tokens(self):
        vila_caption = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups, caption_content='fig', span_group_types=['Caption'])

        vila_caption_dict = FigureTablePredictions._create_dict_of_pages_spans_vila(vila_caption)
        result = self.figure_table_predictor.generate_map_of_layout_to_tokens(
            vila_caption_dict,
            defaultdict(list), defaultdict(list))
        assert list(result.keys()) == []

    def test_predict_1(self):
        result = self.figure_table_predictor.predict()
        assert isinstance(result, dict)
        assert list(result.keys()) == ['figures', 'figure_captions', 'figure_to_figure_captions', 'tables',
                                       'table_captions',
                                       'table_to_table_captions', ]
        assert len(result['figures']) == 4
        assert len(result['tables']) == 4
        assert isinstance(result['figure_to_figure_captions'][0], Relation)
        assert isinstance(result['table_to_table_captions'][0], Relation)
        assert [figure.to_json() for figure in result['figures']] == [{'boxes': [{'height': 0.130624674787425,
                                                                                  'left': 0.5021962683185254,
                                                                                  'page': 0,
                                                                                  'top': 0.3574526237718987,
                                                                                  'width': 0.3930938321780535}]},
                                                                      {'boxes': [{'height': 0.21034525861643782,
                                                                                  'left': 0.08724006952023973,
                                                                                  'page': 2,
                                                                                  'top': 0.09557842485832446,
                                                                                  'width': 0.3754700804068372}],
                                                                       'id': 1},
                                                                      {'boxes': [{'height': 0.31222110318652835,
                                                                                  'left': 0.08188235294117646,
                                                                                  'page': 3,
                                                                                  'top': 0.08723311954074436,
                                                                                  'width': 0.37919526861851516}],
                                                                       'id': 2},
                                                                      {'boxes': [{'height': 0.3527590433756511,
                                                                                  'left': 0.09958468543158637,
                                                                                  'page': 7,
                                                                                  'top': 0.08601251274648339,
                                                                                  'width': 0.8034834020278033}],
                                                                       'id': 3}]
        assert [figure_caption.to_json() for figure_caption in result['figure_captions']] == [
            {'id': 0, 'metadata': {}, 'spans': [{'end': 2057, 'start': 2034}]},
            {'id': 1, 'metadata': {}, 'spans': [{'end': 9679, 'start': 9175}]},
            {'id': 2, 'metadata': {}, 'spans': [{'end': 13875, 'start': 13822}]},
            {'id': 3, 'metadata': {}, 'spans': [{'end': 31364, 'start': 31224}]}]

        assert [table.to_json() for table in result['tables']] == [{'boxes': [{'height': 0.2796805025351168,
                                                                               'left': 0.16789371515411178,
                                                                               'page': 4,
                                                                               'top': 0.1370883614125878,
                                                                               'width': 0.6443845462175756}]},
                                                                   {'boxes': [{'height': 0.20913203075678666,
                                                                               'left': 0.1747694701151131,
                                                                               'page': 5,
                                                                               'top': 0.13721680882001164,
                                                                               'width': 0.622537251391442}],
                                                                    'id': 1},
                                                                   {'boxes': [{'height': 0.06003320096719145,
                                                                               'left': 0.15402431114047183,
                                                                               'page': 5,
                                                                               'top': 0.5840287642045454,
                                                                               'width': 0.2569979998021344}],
                                                                    'id': 2},
                                                                   {'boxes': [{'height': 0.23519277090978136,
                                                                               'left': 0.5027104296715431,
                                                                               'page': 6,
                                                                               'top': 0.27805763784081045,
                                                                               'width': 0.3950077131682751}],
                                                                    'id': 3}]

        assert [table_caption.to_json() for table_caption in result['table_captions']] == [
            {'id': 0, 'metadata': {}, 'spans': [{'end': 18359, 'start': 18198}]},
            {'id': 1, 'metadata': {}, 'spans': [{'end': 22214, 'start': 22042}]},
            {'id': 2, 'metadata': {}, 'spans': [{'end': 23502, 'start': 23400}]},
            {'id': 3, 'metadata': {}, 'spans': [{'end': 29584, 'start': 29369}]}]

    def test_predict_2(self):
        self.figure_table_predictor.doc = self.doc_2
        result = self.figure_table_predictor.predict()
        assert isinstance(result, dict)
        assert list(result.keys()) == ['figures', 'figure_captions', 'figure_to_figure_captions', 'tables',
                                       'table_captions',
                                       'table_to_table_captions', ]
        assert len(result['figures']) == 19
        assert len(result['tables']) == 0
        assert isinstance(result['figure_to_figure_captions'][0], Relation)
        assert [figure.to_json() for figure in result['figures']] == [{'boxes': [{'height': 0.13208454397159292,
                                                                                  'left': 0.09429629629629631,
                                                                                  'page': 1,
                                                                                  'top': 0.12020118024444441,
                                                                                  'width': 0.8439511451193417}]},
                                                                      {'boxes': [{'height': 0.12555849287245008,
                                                                                  'left': 0.10717065442246174,
                                                                                  'page': 2,
                                                                                  'top': 0.12202943166097005,
                                                                                  'width': 0.789011072229456}],
                                                                       'id': 1},
                                                                      {'boxes': [{'height': 0.12768447664048937,
                                                                                  'left': 0.1225524949438778,
                                                                                  'page': 3,
                                                                                  'top': 0.11741933822631836,
                                                                                  'width': 0.7555588459281765}],
                                                                       'id': 2},
                                                                      {'boxes': [{'height': 0.12468946244981555,
                                                                                  'left': 0.26643748342255014,
                                                                                  'page': 4,
                                                                                  'top': 0.27028802235921223,
                                                                                  'width': 0.4831337634428048}],
                                                                       'id': 3},
                                                                      {'boxes': [{'height': 0.630376632095095,
                                                                                  'left': 0.0971906808641975,
                                                                                  'page': 5,
                                                                                  'top': 0.12133209818268287,
                                                                                  'width': 0.7874898394958848}],
                                                                       'id': 4},
                                                                      {'boxes': [{'height': 0.11287960476345482,
                                                                                  'left': 0.6190820387852045,
                                                                                  'page': 5,
                                                                                  'top': 0.6629989624023438,
                                                                                  'width': 0.2615027408050412}],
                                                                       'id': 5},
                                                                      {'boxes': [{'height': 0.17233707639906143,
                                                                                  'left': 0.21671762976626802,
                                                                                  'page': 7,
                                                                                  'top': 0.12308422724405925,
                                                                                  'width': 0.5615297168370628}],
                                                                       'id': 6},
                                                                      {'boxes': [{'height': 0.2378803041246202,
                                                                                  'left': 0.18210305107964408,
                                                                                  'page': 8,
                                                                                  'top': 0.11677710215250652,
                                                                                  'width': 0.6405414848170653}],
                                                                       'id': 7},
                                                                      {'boxes': [{'height': 0.1573001697555555,
                                                                                  'left': 0.1056064937983539,
                                                                                  'page': 9,
                                                                                  'top': 0.23141550496666669,
                                                                                  'width': 0.7980184940930042}],
                                                                       'id': 8},
                                                                      {'boxes': [{'height': 0.12148476109375006,
                                                                                  'left': 0.12457585487230534,
                                                                                  'page': 10,
                                                                                  'top': 0.6850545247395833,
                                                                                  'width': 0.7425578095067189}],
                                                                       'id': 9},
                                                                      {'boxes': [{'height': 0.13544947306315105,
                                                                                  'left': 0.10108373979482141,
                                                                                  'page': 11,
                                                                                  'top': 0.2338108910454644,
                                                                                  'width': 0.7928778879927019}],
                                                                       'id': 10},
                                                                      {'boxes': [{'height': 0.13958769988888886,
                                                                                  'left': 0.09525028428149405,
                                                                                  'page': 12,
                                                                                  'top': 0.1184321415000001,
                                                                                  'width': 0.8139119431176828}],
                                                                       'id': 11},
                                                                      {'boxes': [{'height': 0.1497645378112793,
                                                                                  'left': 0.23651421315385482,
                                                                                  'page': 13,
                                                                                  'top': 0.1186737166510688,
                                                                                  'width': 0.5291386357060185}],
                                                                       'id': 12},
                                                                      {'boxes': [{'height': 0.1738302630036354,
                                                                                  'left': 0.1360490430039143,
                                                                                  'page': 15,
                                                                                  'top': 0.12171450720893012,
                                                                                  'width': 0.760310806604207}],
                                                                       'id': 13},
                                                                      {'boxes': [{'height': 0.17195404900444877,
                                                                                  'left': 0.10790017210407021,
                                                                                  'page': 16,
                                                                                  'top': 0.11612294514973959,
                                                                                  'width': 0.7933600154923804}],
                                                                       'id': 14},
                                                                      {'boxes': [{'height': 0.14378737661973748,
                                                                                  'left': 0.09378395061728395,
                                                                                  'page': 17,
                                                                                  'top': 0.1162407981024848,
                                                                                  'width': 0.8070699744032921}],
                                                                       'id': 15},
                                                                      {'boxes': [{'height': 0.08326610989040799,
                                                                                  'left': 0.3073652918937275,
                                                                                  'page': 18,
                                                                                  'top': 0.11997419993082682,
                                                                                  'width': 0.37918050004621584}],
                                                                       'id': 16},
                                                                      {'boxes': [{'height': 0.1921661376953125,
                                                                                  'left': 0.13613737757804462,
                                                                                  'page': 20,
                                                                                  'top': 0.12010741763644749,
                                                                                  'width': 0.7314195672180428}],
                                                                       'id': 17},
                                                                      {'boxes': [{'height': 0.1091307742344263,
                                                                                  'left': 0.1111559384877113,
                                                                                  'page': 21,
                                                                                  'top': 0.23250712271001817,
                                                                                  'width': 0.573116231088549}],
                                                                       'id': 18}]
        assert [figure_caption.to_json() for figure_caption in result['figure_captions']] == [
            {'id': 0, 'metadata': {}, 'spans': [{'end': 5253, 'start': 5019}]},
            {'id': 1, 'metadata': {}, 'spans': [{'end': 9230, 'start': 8976}]},
            {'id': 2, 'metadata': {}, 'spans': [{'end': 13164, 'start': 12935}]},
            {'id': 3, 'metadata': {}, 'spans': [{'end': 17600, 'start': 17373}]},
            {'id': 4, 'metadata': {}, 'spans': [{'end': 23624, 'start': 23205}]},
            {'id': 5, 'metadata': {}, 'spans': [{'end': 21009, 'start': 20070}]},
            {'id': 6, 'metadata': {}, 'spans': [{'end': 28975, 'start': 28838}]},
            {'id': 7, 'metadata': {}, 'spans': [{'end': 32839, 'start': 32681}]},
            {'id': 8, 'metadata': {}, 'spans': [{'end': 37061, 'start': 36394}]},
            {'id': 9, 'metadata': {}, 'spans': [{'end': 42245, 'start': 42063}]},
            {'id': 10, 'metadata': {}, 'spans': [{'end': 43512, 'start': 43418}]},
            {'id': 11, 'metadata': {}, 'spans': [{'end': 46726, 'start': 46542}]},
            {'id': 12, 'metadata': {}, 'spans': [{'end': 50359, 'start': 50192}]},
            {'id': 13, 'metadata': {}, 'spans': [{'end': 57779, 'start': 57323}]},
            {'id': 14, 'metadata': {}, 'spans': [{'end': 60918, 'start': 60838}]},
            {'id': 15, 'metadata': {}, 'spans': [{'end': 64943, 'start': 64238}]},
            {'id': 16, 'metadata': {}, 'spans': [{'end': 69170, 'start': 68548}]},
            {'id': 17, 'metadata': {}, 'spans': [{'end': 75951, 'start': 75767}]},
            {'id': 18, 'metadata': {}, 'spans': [{'end': 80129, 'start': 79561}]}]
