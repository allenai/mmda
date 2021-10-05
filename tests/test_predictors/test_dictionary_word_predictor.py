"""
Tests for DictionaryWordPredictor

@rauthur
"""

import tempfile
import unittest

from typing import Optional, Set, List


from mmda.predictors.heuristic_predictors.dictionary_word_predictor import (
    DictionaryWordPredictor,
)
from mmda.types.document import Document, SpanGroup
from mmda.types.span import Span


def mock_document(symbols: str, spans: List[Span], rows: List[SpanGroup]) -> Document:
    doc = Document(symbols=symbols)
    doc.annotate(rows=rows)
    doc.annotate(tokens=[SpanGroup(spans=[span]) for span in spans])
    return doc


class TestDictionaryWordPredictor(unittest.TestCase):
    def test_hyphenated_word_combines(self):
        # fmt:off
               #0         10        20        30        40        50        60        70
               #01234567890123456789012345678901234567890123456789012345678901234567890123456789
        text = "The goal of meta-learning is to train a model on a vari-ety of learning tasks"
        # fmt:on

        spans = [
            Span(start=0, end=3),  # The
            Span(start=4, end=8),  # goal
            Span(start=9, end=11),  # of
            Span(start=12, end=25),  # meta-learning (within one row)
            Span(start=26, end=28),  # is
            Span(start=29, end=31),  # to
            Span(start=32, end=37),  # train
            Span(start=38, end=39),  # a
            Span(start=40, end=45),  # model
            Span(start=46, end=48),  # on
            Span(start=49, end=50),  # a
            Span(start=51, end=56),  # vari- (split rows)
            Span(start=56, end=59),  # ety
            Span(start=60, end=62),  # of
            Span(start=63, end=71),  # learning
            Span(start=72, end=77),  # tasks
        ]

        rows = [SpanGroup(spans=spans[0:12]), SpanGroup(spans=spans[12:])]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("variety\n".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor(dictionary_file_path=f.name)
            words = predictor.predict(document)

        self.assertEqual(
            "The goal of meta-learning is to train a model on a variety of learning tasks",
            " ".join([w.text for w in words]),
        )

    def test_hyphenated_nonword_keeps_separate(self):
        # fmt:off
               #0         10        20        30        40        50        60        70
               #01234567890123456789012345678901234567890123456789012345678901234567890123456789
        text = "The goal of meta-learning is to train a model on a vari-ety of learning tasks"
        # fmt:on

        spans = [
            Span(start=0, end=3),  # The
            Span(start=4, end=8),  # goal
            Span(start=9, end=11),  # of
            Span(start=12, end=25),  # meta-learning (within one row)
            Span(start=26, end=28),  # is
            Span(start=29, end=31),  # to
            Span(start=32, end=37),  # train
            Span(start=38, end=39),  # a
            Span(start=40, end=45),  # model
            Span(start=46, end=48),  # on
            Span(start=49, end=50),  # a
            Span(start=51, end=56),  # vari- (split rows)
            Span(start=56, end=59),  # ety
            Span(start=60, end=62),  # of
            Span(start=63, end=71),  # learning
            Span(start=72, end=77),  # tasks
        ]

        rows = [SpanGroup(spans=spans[0:12]), SpanGroup(spans=spans[12:])]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("nothing\n".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor(dictionary_file_path=f.name)
            words = predictor.predict(document)

        self.assertEqual(
            "The goal of meta-learning is to train a model on a vari-ety of learning tasks",
            " ".join([w.text for w in words]),
        )

    def test_local_document_dictionary_adds_words(self):
        # fmt:off
               #0         10        20        30        40        50        60        70
               #01234567890123456789012345678901234567890123456789012345678901234567890123456789
        text = "The goal of TensorFlow is to train models in a Ten-sorFlow manner"
        # fmt:on

        spans = [
            Span(start=0, end=3),  # The
            Span(start=4, end=8),  # goal
            Span(start=9, end=11),  # of
            Span(start=12, end=22),  # Tensorflow (custom word)
            Span(start=23, end=25),  # is
            Span(start=26, end=28),  # to
            Span(start=29, end=34),  # train
            Span(start=35, end=41),  # models
            Span(start=42, end=44),  # in
            Span(start=45, end=46),  # a
            Span(start=47, end=51),  # Ten-
            Span(start=51, end=58),  # sorflow
            Span(start=59, end=65),  # manner
        ]

        rows = [SpanGroup(spans=spans[0:11]), SpanGroup(spans=spans[11:])]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("nothing\n".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor(dictionary_file_path=f.name)
            words = predictor.predict(document)

        self.assertEqual(
            "The goal of TensorFlow is to train models in a TensorFlow manner",
            " ".join([w.text for w in words]),
        )

    #
    def test_optional_plurarl_words_combined(self):
        # fmt:off
               #0         10        20        30        40        50        60        70
               #01234567890123456789012345678901234567890123456789012345678901234567890123456789
        text = "Do you have any up-date(s)? Please share your up-date(s) now."
        # fmt:on

        spans = [
            Span(start=0, end=2),  # Do
            Span(start=3, end=6),  # you
            Span(start=7, end=11),  # have
            Span(start=12, end=15),  # any
            Span(start=16, end=19),  # up-
            Span(start=19, end=27),  # date(s)?
            Span(start=28, end=34),  # Please
            Span(start=35, end=40),  # share
            Span(start=41, end=45),  # your
            Span(start=46, end=49),  # up-
            Span(start=49, end=56),  # date(s)
            Span(start=57, end=61),  # now.
        ]

        rows = [
            SpanGroup(spans=spans[0:5]),
            SpanGroup(spans=spans[5:10]),
            SpanGroup(spans=spans[10:]),
        ]
        document = mock_document(symbols=text, spans=spans, rows=rows)

        with tempfile.NamedTemporaryFile() as f:
            f.write("update\n".encode("utf-8"))
            f.flush()

            predictor = DictionaryWordPredictor(dictionary_file_path=f.name)
            words = predictor.predict(document)

        self.assertEqual(
            "Do you have any update(s)? Please share your update(s) now.",
            " ".join([w.text for w in words]),
        )
