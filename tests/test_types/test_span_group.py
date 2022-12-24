"""
Tests for SpanGroup

@rauthur
"""

import json
import unittest

from mmda.types import Entity, Document, Span


class TestSpanGroup(unittest.TestCase):
    doc: Document

    def setUp(self) -> None:
        self.doc = Document("This is a test document!")

    def test_annotation_attaches_document(self):
        span_group = Entity(id=1, spans=[Span(0, 4), Span(5, 7)])
        self.doc.annotate(tokens=[span_group])

        span_group = self.doc.tokens[0]
        self.assertEqual(["This", "is"], span_group.symbols)



