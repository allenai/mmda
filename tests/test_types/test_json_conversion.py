'''
Description: Test whether all properties for an mmda doc are preserved when
             converting to json and back.
Author:      @soldni

'''

import json
from pathlib import Path

from mmda.types import BoxGroup, SpanGroup, Document, Metadata
from mmda.parsers import PDFPlumberParser


PDFFILEPATH = Path(__file__).parent / "../fixtures/1903.10676.pdf"


def test_span_group_conversion():
    sg = SpanGroup(spans=[], id=3, metadata=Metadata.from_json({"text": "test"}))
    sg2 = SpanGroup.from_json(sg.to_json())
    assert sg2.to_json() == sg.to_json()
    assert sg2.__dict__ == sg.__dict__

    bg = BoxGroup(boxes=[], metadata=Metadata.from_json({"text": "test", "id": 1}))
    bg2 = BoxGroup.from_json(bg.to_json())
    assert bg2.to_json() == bg.to_json()
    assert bg2.__dict__ == bg.__dict__


def test_doc_conversion():
    pdfparser = PDFPlumberParser()

    orig_doc = pdfparser.parse(input_pdf_path=str(PDFFILEPATH))

    json_doc = json.dumps(orig_doc.to_json())
    new_doc = Document.from_json(json.loads(json_doc))

    # We can't just have a `assert new_doc == orig_doc` statement since
    # internal references to the doc itself (e.g. `doc.tokens[0].doc`)
    # would make it fail. instead, we compare specific elements of the doc.

    # compare just token representation and name of fields
    assert orig_doc.symbols == new_doc.symbols
    assert orig_doc.fields == new_doc.fields

    for field_name in orig_doc.fields:
        # this iterates over all span group for this field in both docs
        field_it = zip(
            getattr(orig_doc, field_name),
            getattr(new_doc, field_name)
        )

        # type annotations to keep mypy quiet
        orig_sg: SpanGroup
        new_sg: SpanGroup

        for orig_sg, new_sg in field_it:
            # for each pair, they should have same metadata (type, id,
            # and optionally, text) and same spans.
            assert orig_sg.metadata == new_sg.metadata
            assert orig_sg.spans == new_sg.spans
