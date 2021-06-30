"""

Tests for symbol scraper parser

@kylel


"""

import unittest

from mmda.parsers.symbol_scraper_parser import SymbolScraperParser

class TestSymbolScraperParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sscraper_parser = SymbolScraperParser(sscraper_bin_path='')

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