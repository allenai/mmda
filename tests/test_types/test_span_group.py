"""
Tests for SpanGroup

@rauthur
"""

import json
import unittest

from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.span import Span


class TestSpanGroup(unittest.TestCase):
    doc: Document

    def setUp(self) -> None:
        self.doc = Document("This is a test document!")

    def test_annotation_attaches_document(self):
        span_group = SpanGroup(id=1, spans=[Span(0, 4), Span(5, 7)])
        self.doc.annotate(tokens=[span_group])

        span_group = self.doc.tokens[0]
        self.assertEqual(["This", "is"], span_group.symbols)

    def test_annotation_allows_nesting(self):
        span_group = SpanGroup(id=1, spans=[Span(0, 4), Span(5, 7)])
        nested_span_group = SpanGroup(id=2, spans=[Span(0, 4)], text="This")

        self.doc.annotate(tokens=[span_group])

        span_group = self.doc.tokens[0]
        span_group.annotate(capitalized=[nested_span_group])

        nested_span_group = span_group.capitalized[0]
        self.assertEqual("This", nested_span_group.text)
        self.assertEqual(["This"], nested_span_group.symbols)

    def test_serialization_with_nesting(self):
        span_group = SpanGroup(id=1, spans=[Span(0, 4), Span(5, 7)])
        nested_span_group = SpanGroup(id=2, spans=[Span(0, 4)], text="This")

        self.doc.annotate(tokens=[span_group])

        span_group = self.doc.tokens[0]
        span_group.annotate(capitalized=[nested_span_group])

        json_repr = self.doc.to_json()
        new_doc = Document.from_json(json_repr)

        span_group = new_doc.tokens[0]
        self.assertEqual(["This", "is"], span_group.symbols)

        nested_span_group = span_group.capitalized[0]
        self.assertEqual("This", nested_span_group.text)
        self.assertEqual(["This"], nested_span_group.symbols)
