"""

Tests for Document

@kylel


"""

import unittest

from mmda.types.span import Span
from mmda.types.document import Document, DocSpan, Token, Page, Row, Sent, Block

class TestDocument(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc_json = {
            'text': 'I live in New York. I read the New York Times.',
            'page': [{'start': 0, 'end': 46, 'id': 0}],
            'token': [{'start': 0, 'end': 1, 'id': 0}, {'start': 2, 'end': 6, 'id': 1},
                       {'start': 7, 'end': 9, 'id': 2}, {'start': 10, 'end': 13, 'id': 3},
                       {'start': 14, 'end': 18, 'id': 4}, {'start': 18, 'end': 19, 'id': 5},
                       {'start': 20, 'end': 21, 'id': 6}, {'start': 22, 'end': 26, 'id': 7},
                       {'start': 27, 'end': 30, 'id': 8}, {'start': 31, 'end': 34, 'id': 9},
                       {'start': 35, 'end': 39, 'id': 10}, {'start': 40, 'end': 45, 'id': 11},
                       {'start': 45, 'end': 46, 'id': 12}],
            'row': [],
            'sent': [{'start': 0, 'end': 19, 'id': 0}, {'start': 20, 'end': 46, 'id': 1}],
            'block': []
        }

    def test_from_json(self):
        doc = Document.from_json(self.doc_json)
        assert doc.text == self.doc_json['text']
        assert len(doc.pages) == 1
        assert len(doc.tokens) == 13
        assert len(doc.sents) == 2
        # post-hoc added type to all spans, even if not in Doc JSON
        assert all([page.type == 'page' for page in doc.pages])
        assert all([token.type == 'token' for token in doc.tokens])
        assert all([sent.type == 'sent' for sent in doc.sents])
        # post-hoc added text to all spans, even if not in Doc JSON
        assert all([page.text == doc.text[page.start:page.end] for page in doc.pages])
        assert all([token.text == doc.text[token.start:token.end] for token in doc.tokens])
        assert all([sent.text == doc.text[sent.start:sent.end] for sent in doc.sents])
        # building of indexes between span types works fine
        assert [page.sents for page in doc.pages]
        assert [page.tokens for page in doc.pages]
        assert [sent.pages for sent in doc.sents]
        assert [sent.tokens for sent in doc.sents]
        assert [token.pages for token in doc.tokens]
        assert [token.sents for token in doc.tokens]

    def test_to_json(self):
        doc = Document.from_json(self.doc_json)
        self.assertDictEqual(d1=doc.to_json(), d2=self.doc_json)

    def test_load_spans(self):
        doc = Document.from_json(self.doc_json)
        block_jsons = [{'start': 0, 'end': 19, 'id': 0}, {'start': 20, 'end': 46, 'id': 1}]
        blocks = [
            DocSpan.from_span(span=Span.from_json(span_json=block_json),
                              doc=doc,
                              span_type=Block)
            for block_json in block_jsons
        ]
        doc.load(blocks=blocks)
        # loaded properly
        assert len(doc.blocks) == 2
        # post-hoc added type to all spans, even if not in Doc JSON
        assert all([block.type == Block for block in doc.blocks])
        # post-hoc added text to all spans, even if not in Doc JSON
        assert all([block.text == doc.text[block.start:block.end] for block in doc.blocks])
        # building of indexes between span types works fine
        assert [block.pages for block in doc.blocks]
        assert [block.tokens for block in doc.blocks]
        assert [block.sents for block in doc.blocks]
