"""
Tests for SpacySentencePredictor

@kylel
"""

import tempfile
import unittest
from typing import List, Optional, Set

from mmda.predictors.heuristic_predictors.dictionary_word_predictor import (
    DictionaryWordPredictor,
)
from mmda.types.document import Document, SpanGroup
from mmda.types.span import Span


def mock_document(symbols: str, spans: List[Span], rows: List[SpanGroup]) -> Document:
    doc = Document(symbols=symbols)
    doc.annotate(rows=rows)
    doc.annotate(
        tokens=[SpanGroup(id=i + 1, spans=[span]) for i, span in enumerate(spans)]
    )
    return doc


class TestDictionaryWordPredictor(unittest.TestCase):
    def test_hyphenated_word_combines(self):