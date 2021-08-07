"""

Tests for symbol scraper parser

@kylel


"""

import unittest

from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.types.names import *

class TestSymbolScraperParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sscraper_parser = SymbolScraperParser(sscraper_bin_path='')
        cls.xml = """<Document>
        <runtime>4814</runtime>
        <pagemetrics>
        <page>
                <pagewidth>612.0</pagewidth>
                <pageheight>792.0</pageheight>
                <no>0</no>
                <lines>555</lines>
                <words>888</words>
                <characters>999</characters>
        </page>
        </pagemetrics>
        <Page id="0">
        <Line id="0" BBOX="151.941 678.953 460.06235 678.953">
                <Word id="0">
                        <Char id="0" mergeId="0" BBOX="152.47467 678.953 10.415318 11.63761" RGB = "[0.0]">T</Char>
                        <Char id="1" mergeId="0" BBOX="163.69911 678.953 8.917578 11.63761" RGB = "[0.0]">h</Char>
                        <Char id="2" mergeId="0" BBOX="173.42581 678.712 6.9033756 8.3839" RGB = "[0.0]">e</Char>
                </Word>
                <Word id="1">
                        <Char id="3" mergeId="0" BBOX="185.54544 678.6259 8.228962 12.2401495" RGB = "[0.0]">S</Char>
                        <Char id="4" mergeId="0" BBOX="194.79012 678.953 13.737889 8.142884" RGB = "[0.0]">m</Char>
                        <Char id="5" mergeId="0" BBOX="209.28548 678.712 7.764146 8.3839" RGB = "[0.0]">o</Char>
                        <Char id="6" mergeId="0" BBOX="217.89317 678.712 7.764146 8.3839" RGB = "[0.0]">o</Char>
                        <Char id="7" mergeId="0" BBOX="226.4148 678.7464 5.371205 11.052287" RGB = "[0.0]">t</Char>
                        <Char id="8" mergeId="0" BBOX="232.07866 678.953 8.917578 11.63761" RGB = "[0.0]">h</Char>
                        <Char id="9" mergeId="0" BBOX="241.80536 678.712 6.9033756 8.3839" RGB = "[0.0]">e</Char>
                        <Char id="10" mergeId="0" BBOX="249.44899 678.712 8.762639 11.878627" RGB = "[0.0]">d</Char>
                </Word>
                <Word id="2">
                        <Char id="11" mergeId="0" BBOX="263.16968 678.953 10.053794 11.63761" RGB = "[0.0]">P</Char>
                        <Char id="12" mergeId="0" BBOX="273.49893 678.712 7.764146 8.3839" RGB = "[0.0]">o</Char>
                        <Char id="13" mergeId="0" BBOX="282.10663 678.712 5.7843747 8.3839" RGB = "[0.0]">s</Char>
                        <Char id="14" mergeId="0" BBOX="288.8034 678.712 5.7843747 8.3839" RGB = "[0.0]">s</Char>
                        <Char id="15" mergeId="0" BBOX="295.34525 678.953 4.114481 11.895842" RGB = "[0.0]">i</Char>
                        <Char id="16" mergeId="0" BBOX="300.14835 678.712 8.676562 11.878627" RGB = "[0.0]">b</Char>
                        <Char id="17" mergeId="0" BBOX="309.7029 678.953 4.114481 11.895842" RGB = "[0.0]">i</Char>
                        <Char id="18" mergeId="0" BBOX="314.4888 678.953 4.114481 11.63761" RGB = "[0.0]">l</Char>
                        <Char id="19" mergeId="0" BBOX="319.2747 678.953 4.114481 11.895842" RGB = "[0.0]">i</Char>
                        <Char id="20" mergeId="0" BBOX="324.12943 678.7464 5.371205 11.052287" RGB = "[0.0]">t</Char>
                        <Char id="21" mergeId="0" BBOX="329.7933 675.4238 7.987946 11.465457" RGB = "[0.0]">y</Char>
                </Word>
        </Line>
        <Line id="1" BBOX="223.505 623.184 388.49564 623.184">
                <Word id="3">
                        <Char id="36" mergeId="0" BBOX="223.69429 623.184 6.166849 6.7347174" RGB = "[0.0]">C</Char>
                        <Char id="37" mergeId="0" BBOX="230.30946 623.184 2.3810613 6.884156" RGB = "[0.0]">u</Char>
                        <Char id="38" mergeId="0" BBOX="233.20857 623.184 4.034853 4.71231" RGB = "[0.0]">r</Char>
                        <Char id="39" mergeId="0" BBOX="237.41281 623.04456 4.4931326 4.851786" RGB = "[0.0]">v</Char>
                        <Char id="40" mergeId="0" BBOX="242.35425 623.184 5.160627 4.71231" RGB = "[0.0]">e</Char>
                        <Char id="41" mergeId="0" BBOX="247.96318 621.1317 4.532983 6.764605" RGB = "[0.0]">s</Char>
                </Word>
                <Word id="4">
                        <Char id="42" mergeId="0" BBOX="255.31558 623.184 6.8044558 6.7347174" RGB = "[0.0]">F</Char>
                        <Char id="43" mergeId="0" BBOX="262.50858 623.184 2.3810613 6.884156" RGB = "[0.0]">l</Char>
                        <Char id="44" mergeId="0" BBOX="265.36783 623.04456 4.612684 4.851786" RGB = "[0.0]">o</Char>
                        <Char id="45" mergeId="0" BBOX="270.48862 621.39075 1.8331183 3.337471" RGB = "[0.0]">w</Char>
                </Word>
                <Word id="5">
                        <Char id="46" mergeId="0" BBOX="275.34042 623.184 6.864231 6.7347174" RGB = "[0.0]">R</Char>
                        <Char id="47" mergeId="0" BBOX="282.43378 623.184 5.818158 6.7347174" RGB = "[0.0]">P</Char>
                        <Char id="48" mergeId="0" BBOX="288.5608 623.184 3.4869099 6.7347174" RGB = "[0.0]">I</Char>
                        <Char id="49" mergeId="0" BBOX="292.62555 621.39075 1.8331183 3.337471" RGB = "[0.0]">,</Char>
                </Word>
        </Line>
        </Page>
        </Document>"""
        cls.xml_lines = [line.strip() for line in cls.xml.splitlines()]
        cls.page_to_metrics = {0: {'height': 792.0, 'width': 612.0, 'rows': 555, 'tokens': 888, 'chars': 999}}
        cls.page_to_row_to_tokens = {0: {
            0: [
                {'text': 'The', 'box': [0.24914161764705883, 0.14273611111111115, 0.045513914379084974, 0.014693952020202017, 0]},
                {'text': 'Smoothed', 'box': [0.3031788235294118, 0.14273611111111115, 0.11873560294117647, 0.015867739267676734, 0]},
                {'text': 'Possibility', 'box': [0.4300158169934641, 0.14273611111111115, 0.12191432352941173, 0.018932647727272683, 0]}
            ],
            1: [
                {'text': 'Curves', 'box': [0.365513545751634, 0.2131515151515152, 0.04706188398692812, 0.011132455808080727, 0]},
                {'text': 'Flow', 'box': [0.4171823202614379, 0.2131515151515152, 0.027787840359477123, 0.008692116161616165, 0]},
                {'text': 'RPI,', 'box': [0.44990264705882355, 0.2131515151515152, 0.03123896781045743, 0.008503431060606054, 0]}
            ]}
        }
        cls.doc_json = {
            Symbols: 'The Smoothed Possibility\nCurves Flow RPI,\n',
            Pages: [
                {'_type': 'SpanGroup',
                 'id': 0,
                 'spans': [{'start': 0, 'end': 41, 'box': [0.24914161764705883, 0.14273611111111115, 0.302788522875817, 0.08154785984848478, 0]}]}
            ],
            Tokens: [
                {'_type': 'SpanGroup', 'id': 0, 'spans': [{'start': 0, 'end': 3, 'box': [0.24914161764705883, 0.14273611111111115, 0.045513914379084974, 0.014693952020202017, 0]}]},
                {'_type': 'SpanGroup', 'id': 1, 'spans': [{'start': 4, 'end': 12, 'box': [0.3031788235294118, 0.14273611111111115, 0.11873560294117647, 0.015867739267676734, 0]}]},
                {'_type': 'SpanGroup', 'id': 2, 'spans': [{'start': 13, 'end': 24, 'box': [0.4300158169934641, 0.14273611111111115, 0.12191432352941173, 0.018932647727272683, 0]}]},
                {'_type': 'SpanGroup', 'id': 3, 'spans': [{'start': 25, 'end': 31, 'box': [0.365513545751634, 0.2131515151515152, 0.04706188398692812, 0.011132455808080727, 0]}]},
                {'_type': 'SpanGroup', 'id': 4, 'spans': [{'start': 32, 'end': 36, 'box': [0.4171823202614379, 0.2131515151515152, 0.027787840359477123, 0.008692116161616165, 0]}]},
                {'_type': 'SpanGroup', 'id': 5, 'spans': [{'start': 37, 'end': 41, 'box': [0.44990264705882355, 0.2131515151515152, 0.03123896781045743, 0.008503431060606054, 0]}]}
            ],
            Rows: [
                {'_type': 'SpanGroup', 'id': 0, 'spans': [{'start': 0, 'end': 24, 'box': [0.24914161764705883, 0.14273611111111115, 0.302788522875817, 0.018932647727272683, 0]}]},
                {'_type': 'SpanGroup', 'id': 1, 'spans': [{'start': 25, 'end': 41, 'box': [0.365513545751634, 0.2131515151515152, 0.115628069117647, 0.011132455808080727, 0]}]}
            ]
        }
        # TODO - test the bboxes too

    def test_build_from_sscraper_bbox(self):
        pass

    def test_split_list_by_start_end_tags(self):
        my_list = ['<a>', 'this', '</a>', 'is', 'a', '<a>', 'list', 'of', '</a>', 'tokens']
        starts_ends = self.sscraper_parser._split_list_by_start_end_tags(my_list=my_list, start_tag='<a>', end_tag='</a>')
        assert starts_ends[0] == (0, 3)
        assert starts_ends[1] == (5, 9)

    def test_find_one_and_extract(self):
        my_list = ['this', 'is', 'a', '<a>list</a>', 'of', '<a>tokens</a>']
        result = self.sscraper_parser._find_one_and_extract(my_list=my_list, start_tag='<a>', end_tag='</a>')
        assert result == 'list'

    def test_parse_row_head_tag(self):
        pass

    def test_parse_word_head_tag(self):
        pass

    def test_parse_char_head_tag(self):
        pass

    def test_parse_page_to_metrics(self):
        page_to_metrics = self.sscraper_parser._parse_page_to_metrics(xml_lines=self.xml_lines)
        self.assertDictEqual(d1=page_to_metrics, d2=self.page_to_metrics)

    def test_parse_page_to_row_to_words(self):
        page_to_row_to_words = self.sscraper_parser._parse_page_to_row_to_tokens(xml_lines=self.xml_lines,
                                                                                 page_to_metrics=self.page_to_metrics)
        assert len(page_to_row_to_words) == len(self.page_to_row_to_tokens)
        assert len(page_to_row_to_words[0]) == len(self.page_to_row_to_tokens[0])    # first page
        assert len(page_to_row_to_words[0][0]) == len(self.page_to_row_to_tokens[0][0])  # first row
        assert len(page_to_row_to_words[0][1]) == len(self.page_to_row_to_tokens[0][1])  # second row
        assert page_to_row_to_words[0][0][0]['text'] == self.page_to_row_to_tokens[0][0][0]['text']  # first word
        assert page_to_row_to_words[0][0][1]['text'] == self.page_to_row_to_tokens[0][0][1]['text']  # second word
        assert page_to_row_to_words[0][0][2]['text'] == self.page_to_row_to_tokens[0][0][2]['text']  # third word
        assert page_to_row_to_words[0][1][0]['text'] == self.page_to_row_to_tokens[0][1][0]['text']  # first word
        assert page_to_row_to_words[0][1][1]['text'] == self.page_to_row_to_tokens[0][1][1]['text']  # second word
        assert page_to_row_to_words[0][1][2]['text'] == self.page_to_row_to_tokens[0][1][2]['text']  # third word
        # self.assertDictEqual(d1=page_to_row_to_words, d2=self.page_to_row_to_words)

    def test_convert_nested_text_to_doc_json(self):
        page_to_row_to_tokens = self.sscraper_parser._parse_page_to_row_to_tokens(xml_lines=self.xml_lines,
                                                                                 page_to_metrics=self.page_to_metrics)
        doc_json = self.sscraper_parser._convert_nested_text_to_doc_json(page_to_row_to_tokens=page_to_row_to_tokens)
        from pprint import pprint
        assert doc_json['symbols'] == self.doc_json['symbols']
        for di, dj in zip(doc_json['pages'], self.doc_json['pages']):
            self.assertDictEqual(di, dj)
        for di, dj in zip(doc_json['rows'], self.doc_json['rows']):
            self.assertDictEqual(di, dj)
        for di, dj in zip(doc_json['tokens'], self.doc_json['tokens']):
            self.assertDictEqual(di, dj)

    def test_parse_sscraper_xml(self):
        pass