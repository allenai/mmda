"""

Tests for symbol scraper parser

@kylel


"""

import unittest

from mmda.types.boundingbox import BoundingBox
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser

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
                <lines>2</lines>
                <words>4</words>
                <characters>21</characters>
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
        </Line>
        <Line id="1" BBOX="223.505 623.184 388.49564 623.184">
                <Word id="2">
                        <Char id="36" mergeId="0" BBOX="223.69429 623.184 6.166849 6.7347174" RGB = "[0.0]">C</Char>
                        <Char id="37" mergeId="0" BBOX="230.30946 623.184 2.3810613 6.884156" RGB = "[0.0]">u</Char>
                        <Char id="38" mergeId="0" BBOX="233.20857 623.184 4.034853 4.71231" RGB = "[0.0]">r</Char>
                        <Char id="39" mergeId="0" BBOX="237.41281 623.04456 4.4931326 4.851786" RGB = "[0.0]">v</Char>
                        <Char id="40" mergeId="0" BBOX="242.35425 623.184 5.160627 4.71231" RGB = "[0.0]">e</Char>
                        <Char id="41" mergeId="0" BBOX="247.96318 621.1317 4.532983 6.764605" RGB = "[0.0]">s</Char>
                </Word>
                <Word id="3">
                        <Char id="42" mergeId="0" BBOX="255.31558 623.184 6.8044558 6.7347174" RGB = "[0.0]">F</Char>
                        <Char id="43" mergeId="0" BBOX="262.50858 623.184 2.3810613 6.884156" RGB = "[0.0]">l</Char>
                        <Char id="44" mergeId="0" BBOX="265.36783 623.04456 4.612684 4.851786" RGB = "[0.0]">o</Char>
                        <Char id="45" mergeId="0" BBOX="270.48862 621.39075 1.8331183 3.337471" RGB = "[0.0]">w</Char>
                </Word>
        </Line>
        </Page>
        </Document>"""
        cls.xml_lines = [line.strip() for line in cls.xml.splitlines()]
        cls.page_to_metrics = {0: {'height': 792.0, 'width': 612.0, 'rows': 2, 'words': 4, 'chars': 21}}
        cls.page_to_row_to_words = {0: {0: [{'text': 'The', 'bbox': None}, {'text': 'Smoothed', 'bbox': None}],
                                        1: [{'text': 'Curves', 'bbox': None}, {'text': 'Flow', 'bbox': None}]}}
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
        page_to_row_to_words = self.sscraper_parser._parse_page_to_row_to_words(xml_lines=self.xml_lines,
                                                                                page_to_metrics=self.page_to_metrics)
        assert len(page_to_row_to_words) == len(self.page_to_row_to_words)
        assert len(page_to_row_to_words[0]) == len(self.page_to_row_to_words[0])    # first page
        assert len(page_to_row_to_words[0][0]) == len(self.page_to_row_to_words[0][0])  # first row
        assert len(page_to_row_to_words[0][1]) == len(self.page_to_row_to_words[0][1])  # second row
        assert page_to_row_to_words[0][0][0]['text'] == self.page_to_row_to_words[0][0][0]['text']  # first word
        assert page_to_row_to_words[0][0][1]['text'] == self.page_to_row_to_words[0][0][1]['text']  # second word
        assert page_to_row_to_words[0][1][0]['text'] == self.page_to_row_to_words[0][1][0]['text']  # first word
        assert page_to_row_to_words[0][1][1]['text'] == self.page_to_row_to_words[0][1][1]['text']  # second word
        # self.assertDictEqual(d1=page_to_row_to_words, d2=self.page_to_row_to_words)

    def test_convert_words_to_spans(self):
        words = [{'text': 'The', 'bbox': None}, {'text': 'Smoothed', 'bbox': None}]
        result = self.sscraper_parser._convert_words_to_spans(words=words)
        assert result['text'] == 'The Smoothed'
        spans = result['spans']
        assert spans[0].start == 0
        assert spans[0].end == 3
        assert spans[1].start == 4
        assert spans[1].end == 12

