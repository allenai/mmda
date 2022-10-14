'''
Description: Test whether all properties for an mmda doc are preserved when
             converting to json and back.
Author:      @soldni

'''

import json
from pathlib import Path

from mmda.types import BoxGroup, SpanGroup, Document, Metadata, Relation
from mmda.parsers import PDFPlumberParser

PDFFILEPATH = Path(__file__).parent / "../fixtures/1903.10676.pdf"


def test_span_group_conversion():
    sg = SpanGroup(spans=[], id=3, metadata=Metadata(text='test'))
    sg2 = SpanGroup.from_json(sg.to_json())
    assert sg2.to_json() == sg.to_json()
    assert sg2.__dict__ == sg.__dict__

    bg = BoxGroup(boxes=[], metadata=Metadata(text='test'))
    bg2 = BoxGroup.from_json(bg.to_json())
    assert bg2.to_json() == bg.to_json()
    assert bg2.__dict__ == bg.__dict__


def test_relation_conversion():
    r = Relation(
        key=SpanGroup(spans=[], id=3, metadata=Metadata(foobar='test'), field='abc'),
        value=SpanGroup(spans=[], id=5, metadata=Metadata(foobar='test'), field='xyz'),
        id=999,
        metadata=Metadata(blabla='something')
    )

    # minimal to & from JSON (default behavior)
    r_dict_minimal = {
        'key': 'abc-3',
        'value': 'xyz-5',
        'id': 999,
        'metadata': {'blabla': 'something'}
    }
    assert r.to_json() == r.to_json(is_minimal=True) == r_dict_minimal

    doc = Document.from_json(doc_dict={
        'symbols': 'asdfasdf',
        'abc': [{'spans': [], 'id': 3, 'metadata': {'foobar': 'test'}}],
        'xyz': [{'spans': [], 'id': 5, 'metadata': {'foobar': 'test'}}]
    })
    assert r_dict_minimal == r.from_json(r_dict_minimal, is_minimal=True, doc=doc).to_json() == \
           r.from_json(r_dict_minimal, doc=doc).to_json()

    # full to JSON
    r_dict_full = {
        'key': {'spans': [], 'id': 3, 'metadata': {'foobar': 'test'}},
        'value': {'spans': [], 'id': 5, 'metadata': {'foobar': 'test'}},
        'id': 999,
        'metadata': {'blabla': 'something'}
    }
    assert r.to_json(is_minimal=False) == r_dict_full


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
