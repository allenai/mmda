"""


@kylel

"""

import json
import pathlib
import unittest

from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.span import Span
from mmda.utils.stringify import stringify_span_group


class TestStringify(unittest.TestCase):
    def test_stringify(self):
        doc = Document.from_json(
            {
                "symbols": '[1] Alan Higgins and R Wohlford,\n"Keyword recognition, "in Proc.   ICASSP , 1990, pp. 1233– 1236.',
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 3}], "text": "[1]"},
                    {"id": 1, "spans": [{"start": 4, "end": 8}], "text": "Alan"},
                    {"id": 2, "spans": [{"start": 9, "end": 16}], "text": "Higgins"},
                    {"id": 3, "spans": [{"start": 17, "end": 20}], "text": "and"},
                    {"id": 4, "spans": [{"start": 21, "end": 22}], "text": "R"},
                    {"id": 5, "spans": [{"start": 23, "end": 31}], "text": "Wohlford"},
                    {"id": 6, "spans": [{"start": 31, "end": 32}], "text": ","},
                    {"id": 7, "spans": [{"start": 33, "end": 34}], "text": '"'},
                    {"id": 8, "spans": [{"start": 34, "end": 41}], "text": "Keyword"},
                    {
                        "id": 9,
                        "spans": [{"start": 42, "end": 53}],
                        "text": "recognition",
                    },
                    {"id": 10, "spans": [{"start": 53, "end": 54}], "text": ","},
                    {"id": 11, "spans": [{"start": 55, "end": 56}], "text": '"'},
                    {"id": 12, "spans": [{"start": 56, "end": 58}], "text": "in"},
                    {"id": 13, "spans": [{"start": 59, "end": 63}], "text": "Proc"},
                    {"id": 14, "spans": [{"start": 63, "end": 64}], "text": "."},
                    {"id": 15, "spans": [{"start": 67, "end": 73}], "text": "ICASSP"},
                    {"id": 16, "spans": [{"start": 74, "end": 75}], "text": ","},
                    {"id": 17, "spans": [{"start": 76, "end": 80}], "text": "1990"},
                    {"id": 18, "spans": [{"start": 80, "end": 81}], "text": ","},
                    {"id": 19, "spans": [{"start": 82, "end": 85}], "text": "pp."},
                    {"id": 20, "spans": [{"start": 86, "end": 90}], "text": "1233"},
                    {"id": 21, "spans": [{"start": 90, "end": 91}], "text": "–"},
                    {"id": 22, "spans": [{"start": 92, "end": 96}], "text": "1236"},
                    {"id": 23, "spans": [{"start": 96, "end": 97}], "text": "."},
                ],
            }
        )
        # make sure test fixture is defined correctly
        for word in doc.words:
            assert word.text == doc.symbols[word.start : word.end]

        # SpanGroup with single span
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 16},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "[1] Alan Higgins",
        )

        # SpanGroup with multiple spans
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 16},
                    {"start": 17, "end": 20},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "[1] Alan Higgins and",
        )

        # SpanGroup with disjoint spans -> grabs symbols & join, but ignore text in-between
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 16},
                    {"start": 23, "end": 31},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "[1] Alan Higgins Wohlford",
        )

        # stringify the whole bib entry
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 97},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            '[1] Alan Higgins and R Wohlford, "Keyword recognition, "in Proc. ICASSP , 1990, pp. 1233– 1236.',
        )

    def test_multiple_whitespace(self):
        doc = Document.from_json(
            {
                "symbols": "This is a   \n     test.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 4}], "text": "This"},
                    {"id": 1, "spans": [{"start": 5, "end": 7}], "text": "is"},
                    {"id": 2, "spans": [{"start": 8, "end": 9}], "text": "a"},
                    {"id": 3, "spans": [{"start": 18, "end": 22}], "text": "test"},
                    {"id": 4, "spans": [{"start": 22, "end": 23}], "text": "."},
                ],
            }
        )
        # make sure test fixture is defined correctly
        for word in doc.words:
            assert word.text == doc.symbols[word.start : word.end]

        # does whitespace normalize fine?
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 24},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "This is a test.",
        )

    def test_partial_word_match(self):
        doc = Document.from_json(
            {
                "symbols": "This is a test.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 4}], "text": "This"},
                    {"id": 1, "spans": [{"start": 5, "end": 7}], "text": "is"},
                    {"id": 2, "spans": [{"start": 8, "end": 9}], "text": "a"},
                    {"id": 3, "spans": [{"start": 10, "end": 14}], "text": "test"},
                    {"id": 4, "spans": [{"start": 14, "end": 15}], "text": "."},
                ],
            }
        )
        # make sure test fixture is defined correctly
        for word in doc.words:
            assert word.text == doc.symbols[word.start : word.end]

        # does it grab partial word matches?
        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 2, "end": 7},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "This is",
        )

        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 6, "end": 13},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "is a test",
        )

    def test_use_word_metadata_text(self):
        doc = Document.from_json(
            {
                "symbols": "This is a te-\nst.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 4}], "text": "This"},
                    {"id": 1, "spans": [{"start": 5, "end": 7}], "text": "is"},
                    {"id": 2, "spans": [{"start": 8, "end": 9}], "text": "a"},
                    {"id": 3, "spans": [{"start": 10, "end": 16}], "text": "test"},
                    {"id": 4, "spans": [{"start": 16, "end": 17}], "text": "."},
                ],
            }
        )
        # make sure test fixture is defined correctly
        for i, word in enumerate(doc.words):
            if i != 3:
                assert word.text == doc.symbols[word.start : word.end]
            else:
                assert word.text == "test"
                assert doc.symbols[word.start : word.end] == "te-\nst"

        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 5, "end": 17},
                ]
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "is a test.",
        )

    def test_how_words_relate_to_stringify(self):
        """This test is a comprehensive dive into how `words` interacts
        with `stringify()`. There are 4 cases defined here:

        1. `words` arent comprehensive. That is, `doc.symbols` contains chars
           that arent whitespace but also not part of any `word.

        2. `words` are comprehensive. each word is effectively a token.

        3. `words` are comprehensive. each word is a bigger chunk (includes punct)

        4. `words` are comprehensive and big chunks. they also override the text.

        """

        # 1) for example, what might happen is puncts and newlines aren't included in words
        doc = Document.from_json(
            {
                "symbols": "Symbols in-\nclude hyph- ens.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 7}], "text": "Symbols"},
                    {"id": 1, "spans": [{"start": 8, "end": 10}], "text": "in"},
                    {"id": 2, "spans": [{"start": 12, "end": 17}], "text": "clude"},
                    {"id": 3, "spans": [{"start": 18, "end": 22}], "text": "hyph"},
                    {"id": 4, "spans": [{"start": 24, "end": 27}], "text": "ens"},
                ],
            }
        )
        # make sure test fixture is defined correctly
        for i, word in enumerate(doc.words):
            assert word.text == doc.symbols[word.start : word.end]

        query_span_group = SpanGroup.from_json(
            {
                "spans": [
                    {"start": 0, "end": 28},
                ]
            }
        )

        # intended behavior here is that default should return just
        # the strings from the `word.text`, stitched together by whitespace
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "Symbols in clude hyph ens",
        )

        # 2) now repeat this test, but keeping the punctuation as indiv words
        doc = Document.from_json(
            {
                "symbols": "Symbols in-\nclude hyph- ens.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 7}], "text": "Symbols"},
                    {"id": 1, "spans": [{"start": 8, "end": 10}], "text": "in"},
                    {"id": 2, "spans": [{"start": 10, "end": 11}], "text": "-"},
                    {"id": 3, "spans": [{"start": 12, "end": 17}], "text": "clude"},
                    {"id": 4, "spans": [{"start": 18, "end": 22}], "text": "hyph"},
                    {"id": 5, "spans": [{"start": 22, "end": 23}], "text": "-"},
                    {"id": 6, "spans": [{"start": 24, "end": 27}], "text": "ens"},
                    {"id": 7, "spans": [{"start": 27, "end": 28}], "text": "."},
                ],
            }
        )
        for i, word in enumerate(doc.words):
            assert word.text == doc.symbols[word.start : word.end]

        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "Symbols in- clude hyph- ens.",
        )

        # 3) repeat this test, but merging hyphen into bigger word chunks
        doc = Document.from_json(
            {
                "symbols": "Symbols in-\nclude hyph- ens.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 7}], "text": "Symbols"},
                    {"id": 1, "spans": [{"start": 8, "end": 11}], "text": "in-"},
                    {"id": 2, "spans": [{"start": 12, "end": 17}], "text": "clude"},
                    {"id": 3, "spans": [{"start": 18, "end": 23}], "text": "hyph-"},
                    {"id": 4, "spans": [{"start": 24, "end": 28}], "text": "ens."},
                ],
            }
        )
        for i, word in enumerate(doc.words):
            assert word.text == doc.symbols[word.start : word.end]

        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "Symbols in- clude hyph- ens.",
        )

        # 4) finally, let's override the word text with alternative string
        doc = Document.from_json(
            {
                "symbols": "Symbols in-\nclude hyph- ens.",
                "words": [
                    {"id": 0, "spans": [{"start": 0, "end": 7}], "text": "Symbols"},
                    {"id": 1, "spans": [{"start": 8, "end": 11}], "text": "IN"},
                    {"id": 2, "spans": [{"start": 12, "end": 17}], "text": "clude"},
                    {"id": 3, "spans": [{"start": 18, "end": 23}], "text": "HYPH"},
                    {"id": 4, "spans": [{"start": 24, "end": 28}], "text": "ENS"},
                ],
            }
        )
        self.assertEqual(
            stringify_span_group(span_group=query_span_group, document=doc),
            "Symbols IN clude HYPH ENS",
        )
