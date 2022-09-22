"""

@kylel

"""


import tempfile
import unittest
from typing import List, Optional, Set

from mmda.predictors.heuristic_predictors.bib_page_detector import BibPagePredictor
from mmda.types.document import Document, SpanGroup
from mmda.types.span import Span


class TestBibPagePredictor(unittest.TestCase):
    def setUp(self):
        self.doc = Document(symbols='This is a test docment.')

    def test_predict(self):
        pass




