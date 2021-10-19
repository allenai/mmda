import unittest

from mmda.eval.metrics import levenshtein


class TestLevenshteinDistance(unittest.TestCase):
    def test_calculates(self):
        assert levenshtein("hello", "kelm") == 3
        assert levenshtein("kelm", "hello") == 3

        assert levenshtein("", "hello") == 5
        assert levenshtein("heck", "hecko") == 1
        assert levenshtein("ecko", "hecko") == 1

    def test_unicode_edits(self):
        assert levenshtein("Na+", "Na\u207a") == 1

    def test_case_sensitivity(self):
        assert levenshtein("Hello", "heLlo") == 2
        assert levenshtein("Hello", "heLlo", case_sensitive=False) == 0
