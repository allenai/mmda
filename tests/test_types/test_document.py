import json
import unittest
import os

from mmda.types.document import Document
from mmda.types.names import MetadataField, SymbolsField

from ai2_internal import api


def resolve(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), "../fixtures", file)


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

    def test_annotate_box_groups_gets_text(self):
        # when token boxes are on box_groups
        spp_plumber_doc = "spp-dag-0-0-4-doc.json"
        doc_file = resolve(spp_plumber_doc)
        with open(doc_file) as f:
            spp_doc = Document.from_json(json.load(f))

        with open(resolve("test_document_box_groups.json")) as f:
            box_groups = [api.BoxGroup(**bg).to_mmda() for bg in json.load(f)["grobid_bibs_box_groups"]]

        spp_doc.annotate(new_span_groups=box_groups)
        assert spp_doc.new_span_groups[0].text.startswith("Gutman G, Rosenzweig D, Golan J")

        # when token boxes are on spans
        plumber_doc = "c8b53e2d9cd247e2d42719e337bfb13784d22bd2.json"
        doc_file = resolve(plumber_doc)
        with open(doc_file) as f:
            doc = Document.from_json(json.load(f))

        with open(resolve("test_document_box_groups.json")) as f:
            box_groups = [api.BoxGroup(**bg).to_mmda() for bg in json.load(f)["grobid_bibs_box_groups"]]

        doc.annotate(new_span_groups=box_groups)
        assert doc.new_span_groups[0].text.startswith("Gutman G, Rosenzweig D, Golan J")
