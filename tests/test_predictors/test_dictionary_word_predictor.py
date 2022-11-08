"""
Tests for DictionaryWordPredictor

@rauthur
"""

import tempfile
import unittest
from typing import List, Optional, Set

from mmda.predictors.heuristic_predictors.dictionary_word_predictor import Dictionary
from mmda.predictors import DictionaryWordPredictor
from mmda.types import Document, SpanGroup, Span


def mock_document(symbols: str, spans: List[Span], rows: List[SpanGroup]) -> Document:
    doc = Document(symbols=symbols)
    doc.annotate(rows=rows)
    doc.annotate(
        tokens=[SpanGroup(id=i, spans=[span]) for i, span in enumerate(spans)]
    )
    return doc


# 'fine-tuning', 'sentence', 'word'
class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.dict = Dictionary(words=[], punct='-!%')

    def test_add_and_is_in(self):
        self.dict.add('Fine-TuninG')
        self.dict.add('--fine-tuning--')
        self.assertTrue(self.dict.is_in('Fine-TuninG'))
        self.assertTrue(self.dict.is_in('--fine-tuning--'))
        self.assertTrue(self.dict.is_in('---FINE-TUNING----'))
        self.assertTrue(self.dict.is_in('fine-tuning'))
        self.assertFalse(self.dict.is_in('fine'))
        self.assertFalse(self.dict.is_in('tuning'))
        self.assertFalse(self.dict.is_in('finetuning'))

    def test_strip_punct(self):
        self.assertEqual(self.dict.strip_punct(text='fine-tuning'), 'fine-tuning')
        self.assertEqual(self.dict.strip_punct(text='123fine-tuning123'), '123fine-tuning123')
        self.assertEqual(self.dict.strip_punct(text='--fine-tuning--'), 'fine-tuning')
        self.assertEqual(self.dict.strip_punct(text='!!--fine-tuning--!!'), 'fine-tuning')
        self.assertEqual(self.dict.strip_punct(text='%!!--fine-tuning--!!%'), 'fine-tuning')
        # because # is not part of the dictionary, stops stripping
        self.assertEqual(self.dict.strip_punct(text='#--fine-tuning--#'), '#--fine-tuning--#')
        self.assertEqual(self.dict.strip_punct(text='--#--fine-tuning--#--'), '#--fine-tuning--#')


class TestDictionaryWordPredictor(unittest.TestCase):
    def test_hyphenated_word_combines(self):
        # fmt:off
        # 0         10        20        30        40        50        60        70
        # 01234567890123456789012345678901234567890123456789012345678901234567890123456789
        text = "The goal of meta-learning is to train a model on a vari-\nety! of learning tasks."
        # fmt:on

        spans = [
            Span(start=0, end=3),  # The
            Span(start=4, end=8),  # goal
            Span(start=9, end=11),  # of
            Span(start=12, end=16),  # meta
            Span(start=16, end=17),  # -
            Span(start=17, end=25),  # learning
            Span(start=26, end=28),  # is
            Span(start=29, end=31),  # to
            Span(start=32, end=37),  # train
            Span(start=38, end=39),  # a
            Span(start=40, end=45),  # model
            Span(start=46, end=48),  # on
            Span(start=49, end=50),  # a
            Span(start=51, end=55),  # vari
            Span(start=55, end=56),  # -
            Span(start=57, end=60),  # ety
            Span(start=60, end=61),  # !
            Span(start=62, end=64),  # of
            Span(start=65, end=73),  # learning
            Span(start=74, end=79),  # tasks
            Span(start=79, end=80),  # .
        ]

        rows = [SpanGroup(id=0, spans=spans[0:15]), SpanGroup(id=1, spans=spans[15:])]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("variety\n".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor()
            words = predictor.predict(document)
            document.annotate(words=words)

        self.assertEqual(
            [w.text for w in words],
            ['The', 'goal', 'of', 'meta-learning', 'is', 'to', 'train', 'a', 'model',
             'on', 'a', 'vari-ety', '!', 'of', 'learning', 'tasks', '.'])


    def test_next_row_single_token(self):
        # fmt:off
        # 0         10
        # 012345678901
        text = "Many lin-es"
        # fmt:on

        spans = [
            Span(start=0, end=4),  # Many
            Span(start=5, end=9),  # lin-
            Span(start=9, end=11),  # es
        ]

        rows = [
            SpanGroup(id=1, spans=spans[0:2]),
            SpanGroup(id=2, spans=spans[2:3]),
        ]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor(dictionary_file_path=f.name)
            words = predictor.predict(document)

        self.assertEqual([w.text for w in words], ['Many', 'lin-es'])
