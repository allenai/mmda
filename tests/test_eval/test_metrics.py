import unittest

from mmda.eval.metrics import box_overlap, levenshtein
from mmda.types.box import Box


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

    def test_strips_spaces(self):
        assert levenshtein("\nHel lo\r", "Hello") == 3
        assert levenshtein(" Hel lo ", "Hello", strip_spaces=True) == 0

    def test_normalizes(self):
        assert levenshtein("\nHel lo\r", "Hello") == 3
        assert levenshtein(" Hel lo ", "Hello", normalize=True) == 0.375


class TestBoxOverlap(unittest.TestCase):
    def _box(self, l, t, w, h):
        return Box(l=l, t=t, w=w, h=h, page=0)

    def test_consumed(self):
        box = self._box(1.0, 2.0, 3.0, 3.0)
        container = self._box(0.0, 0.0, 4.0, 5.0)

        assert box_overlap(box, container) == 1.0

    def test_no_overlap(self):
        box = self._box(0.0, 0.0, 1.0, 1.0)
        container = self._box(2.0, 2.0, 1.0, 1.0)

        assert box_overlap(box, container) == 0.0

    def test_partially_contained_top(self):
        box = self._box(1.0, 0.0, 1.0, 2.0)
        container = self._box(0.0, 1.0, 100.0, 2.0)

        assert box_overlap(box, container) == 0.5
        assert box_overlap(container, box) == 1.0 / 200.0

    def test_partially_contained_bottom(self):
        box = self._box(1.0, 1.0, 1.0, 2.0)
        container = self._box(0.0, 0.0, 100.0, 2.0)

        assert box_overlap(box, container) == 0.5
        assert box_overlap(container, box) == 1.0 / 200.0

    def test_partially_contained_left(self):
        box = self._box(0.0, 2.0, 2.0, 1.0)
        container = self._box(1.0, 1.0, 2.0, 100.0)

        assert box_overlap(box, container) == 0.5
        assert box_overlap(container, box) == 1.0 / 200.0

    def test_partially_contained_right(self):
        box = self._box(1.0, 1.0, 2.0, 1.0)
        container = self._box(0.0, 0.0, 2.0, 100.0)

        assert box_overlap(box, container) == 0.5
        assert box_overlap(container, box) == 1.0 / 200.0

    def test_partially_contained_corner(self):
        box = self._box(1.0, 0.0, 2.0, 2.0)
        container = self._box(0.0, 1.0, 2.0, 2.0)

        assert box_overlap(box, container) == 0.25
