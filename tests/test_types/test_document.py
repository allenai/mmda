import unittest

from mmda.types.document import Document
from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.span import Span
from mmda.types.box import Box
from mmda.types.names import MetadataField, SymbolsField, TokensField, PagesField


class TestDocument(unittest.TestCase):
    def test__empty_annotations_work(self):
        doc = Document("This is a test document!")
        annotations = []
        doc.annotate(my_cool_field=annotations)
        self.assertEqual(doc.my_cool_field, [])

    def test_metadata_serializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey there y'all!"
        doc = Document(symbols=symbols)
        doc.add_metadata(**metadata)

        output_json = doc.to_json()
        self.assertDictEqual(
            {SymbolsField: symbols, MetadataField: metadata}, output_json
        )

    def test_metadata_deserializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey again peeps!"
        input_json = {SymbolsField: symbols, MetadataField: metadata}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertDictEqual(metadata, doc.metadata.to_json())

    def test_metadata_deserializes_when_empty(self):
        symbols = "That's all folks!"
        input_json = {SymbolsField: symbols}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertEqual(0, len(doc.metadata))

    def test_box_group_annotation(self):
        symbols = "Hey I am Doc!"
        tokens = [
            SpanGroup(
                spans=[
                    Span(start=0, end=3, box=Box(l=0.1, t=0.1, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=4, end=5, box=Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=6, end=8, box=Box(l=0.5, t=0.5, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=9, end=13, box=Box(l=0.7, t=0.7, w=0.1, h=0.1, page=0))
                ]
            ),
        ]
        pages = [
            SpanGroup(
                spans=[
                    Span(
                        start=0,
                        end=13,
                        box=Box.small_boxes_to_big_box(
                            [t.spans[0].box for t in tokens]
                        ),
                    )
                ]
            )
        ]

        input_json = {
            SymbolsField: symbols,
            TokensField: [t.to_json() for t in tokens],
            PagesField: [p.to_json() for p in pages],
        }

        box_group = BoxGroup(
            boxes=[
                Box(l=0.1, t=0.1, w=0.9, h=0.9, page=0),
                Box(l=0.5, t=0.5, w=0.3, h=0.3, page=0),
            ]
        )

        doc = Document.from_json(input_json)
        doc.annotate(boxed=[box_group])

        self.assertEqual(doc.symbols, " ".join([t.text for t in doc.boxed[0].tokens]))
        self.assertEqual(1, len(doc.boxed[0].spans))

    def test_box_group_annotation_multipage(self):
        symbols = "Hey I am Doc!"
        tokens = [
            SpanGroup(
                spans=[
                    Span(start=0, end=3, box=Box(l=0.1, t=0.1, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=4, end=5, box=Box(l=0.3, t=0.3, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=6, end=8, box=Box(l=0.5, t=0.5, w=0.1, h=0.1, page=0))
                ]
            ),
            SpanGroup(
                spans=[
                    Span(start=9, end=13, box=Box(l=0.3, t=0.3, w=0.1, h=0.1, page=1))
                ]
            ),
        ]
        pages = [
            SpanGroup(
                spans=[
                    Span(
                        start=0,
                        end=8,
                        box=Box.small_boxes_to_big_box(
                            [t.spans[0].box for t in tokens[:3]]
                        ),
                    )
                ]
            ),
            SpanGroup(
                spans=[
                    Span(
                        start=9,
                        end=13,
                        box=Box.small_boxes_to_big_box(
                            [t.spans[0].box for t in tokens[-1:]]
                        ),
                    )
                ]
            ),
        ]

        input_json = {
            SymbolsField: symbols,
            TokensField: [t.to_json() for t in tokens],
            PagesField: [p.to_json() for p in pages],
        }

        box_group = BoxGroup(
            boxes=[
                Box(l=0.1, t=0.1, w=0.9, h=0.9, page=0),
                Box(l=0.1, t=0.1, w=0.9, h=0.9, page=1),
            ]
        )

        doc = Document.from_json(input_json)
        doc.annotate(boxed=[box_group])

        # Output still the same as previous test but multiple spans in span group
        self.assertEqual(doc.symbols, " ".join([t.text for t in doc.boxed[0].tokens]))
        self.assertEqual(2, len(doc.boxed[0].spans))
