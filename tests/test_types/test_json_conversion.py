'''
Description: Test whether all properties for an mmda doc are preserved when
             converting to json and back.
Author:      @soldni, @kylel

'''

import unittest
import json

from mmda.types import Span, Box, BoxGroup, SpanGroup, Document, Metadata, Relation


class TestJSONConversion(unittest.TestCase):
    def setUp(self):
        self.doc_dict = {'symbols': 'a b c d e f g'}
        self.sg_dict = {'spans': [{'start': 0, 'end': 2}], 'id': 3}
        self.sg2_dict = {'spans': [{'start': 4, 'end': 6}], 'id': 5}
        self.meta_dict = {'foobar': 'test'}

    def test_boxes(self):
        # minimal span
        b = Box(l=0.0, t=0.1, w=0.2, h=0.3, page=4)
        b_dict = {'left': 0.0, 'top': 0.1, 'width': 0.2, 'height': 0.3, 'page': 4}
        assert b.to_json() == b_dict
        assert Box.from_json(box_dict=b_dict).to_json() == b_dict

    def test_spans(self):
        # minimal span
        s = Span(start=0, end=2)
        s_dict = {'start': 0, 'end': 2}
        assert s.to_json() == s_dict
        assert Span.from_json(span_dict=s_dict).to_json() == s_dict

        # contains boxes
        s = Span(start=0, end=2, box=Box(l=0.0, t=0.1, w=0.2, h=0.3, page=4))
        s_dict = {'start': 0, 'end': 2,
                  'box': {'left': 0.0, 'top': 0.1, 'width': 0.2, 'height': 0.3, 'page': 4}}
        assert s.to_json() == s_dict
        assert Span.from_json(span_dict=s_dict).to_json() == s_dict

    def test_metadata(self):
        # empty metadata
        m = Metadata()
        m_dict = {}
        assert m.to_json() == m_dict
        assert Metadata.from_json(di=m_dict).to_json() == m_dict

        # null-valued metadata
        m = Metadata(foo=None, bar=None)
        m_dict = {'foo': None, 'bar': None}
        assert m.to_json() == m_dict
        assert Metadata.from_json(di=m_dict).to_json() == m_dict

        # meaningful metadata
        m = Metadata(foo='xyz', bar='abc')
        m_dict = {'foo': 'xyz', 'bar': 'abc'}
        assert m.to_json() == m_dict
        assert Metadata.from_json(di=m_dict).to_json() == m_dict

    def test_box_groups(self):
        # minimal box group
        bg = BoxGroup(boxes=[])
        bg_dict = {'boxes': []}
        assert bg.to_json() == bg_dict
        assert BoxGroup.from_json(box_group_dict=bg_dict).to_json() == bg_dict

        # slightly more stuff in box group
        bg = BoxGroup(boxes=[], id=999, doc=Document(symbols='doesnt-matter-what-goes-here'),
                      field='also-ignored', metadata=Metadata(foo='bar'))
        bg_dict = {'boxes': [], 'id': 999, 'metadata': {'foo': 'bar'}}
        assert bg.to_json() == bg_dict
        assert BoxGroup.from_json(box_group_dict=bg_dict).to_json() == bg_dict

        # add boxes to boxgroup
        bg = BoxGroup(boxes=[Box(l=0.0, t=0.1, w=0.2, h=0.3, page=4),
                             Box(l=0.5, t=0.6, w=0.7, h=0.8, page=9)])
        bg_dict = {'boxes': [{'left': 0.0, 'top': 0.1, 'width': 0.2, 'height': 0.3, 'page': 4},
                             {'left': 0.5, 'top': 0.6, 'width': 0.7, 'height': 0.8, 'page': 9}]}
        assert bg.to_json() == bg_dict
        assert BoxGroup.from_json(box_group_dict=bg_dict).to_json() == bg_dict

    def test_span_groups(self):
        # minimal span group
        sg = SpanGroup(spans=[])
        sg_dict = {'spans': []}
        assert sg.to_json() == sg_dict
        assert SpanGroup.from_json(span_group_dict=sg_dict).to_json() == sg_dict

        # slightly more stuff in span group
        sg = SpanGroup(spans=[], id=999, doc=Document(symbols='doesnt-matter-what-goes-here'),
                       field='also-ignored', metadata=Metadata(foo='bar'))
        sg_dict = {'spans': [], 'id': 999, 'metadata': {'foo': 'bar'}}
        assert sg.to_json() == sg_dict
        assert SpanGroup.from_json(span_group_dict=sg_dict).to_json() == sg_dict

        # add spans to spangroup
        sg = SpanGroup(spans=[Span(start=0, end=2), Span(start=3, end=4)])
        sg_dict = {'spans': [{'start': 0, 'end': 2}, {'start': 3, 'end': 4}]}
        assert sg.to_json() == sg_dict
        assert SpanGroup.from_json(span_group_dict=sg_dict).to_json() == sg_dict

        # contains boxgroup
        sg = SpanGroup(spans=[], box_group=BoxGroup(boxes=[]))
        sg_dict = {'spans': [], 'box_group': {'boxes': []}}
        assert sg.to_json() == sg_dict
        assert SpanGroup.from_json(span_group_dict=sg_dict).to_json() == sg_dict

    def test_documents(self):
        # minimal doc
        doc = Document(symbols='a b c d e f g')
        doc_dict = {'symbols': 'a b c d e f g'}
        assert doc.to_json() == doc_dict
        assert Document.from_json(doc_dict=doc_dict).to_json() == doc_dict

        # doc with span group
        doc_dict = {'symbols': 'a b c d e f g', 'stuff': [{'spans': []}]}
        doc = Document.from_json(doc_dict)
        assert doc.fields == ['stuff']  # from_json() should apply .annotation()
        assert len(doc.stuff) == 1
        assert doc.stuff[0].to_json() == {'spans': []}
        assert doc.to_json() == doc_dict

    def test_relations(self):
        # minimal relation still requires SpanGroup to have names
        with self.assertRaises(ValueError):
            Relation(key=SpanGroup(spans=[]), value=SpanGroup(spans=[]))

        # minimal relation working example
        sg1 = SpanGroup(spans=[], id=123, field='abc')
        sg2 = SpanGroup(spans=[], id=999, field='xyz')
        r = Relation(key=sg1, value=sg2)
        r_dict = {'key': 'abc-123', 'value': 'xyz-999'}
        assert r.to_json() == r_dict

        # to test `from_json()` we need a Document annotated w the related units
        doc = Document(symbols='a b c d e f g')
        sg1 = SpanGroup(spans=[], id=123)
        sg2 = SpanGroup(spans=[], id=999)
        doc.annotate(abc=[sg1])
        doc.annotate(xyz=[sg2])
        r_dict = {'key': 'abc-123', 'value': 'xyz-999'}
        assert Relation.from_json(relation_dict=r_dict, doc=doc).to_json() == r_dict

        # `from_json()` should fail if Document isnt coherent with fieldnames
        doc = Document(symbols='a b c d e f g')
        sg1 = SpanGroup(spans=[], id=123)
        sg2 = SpanGroup(spans=[], id=999)
        doc.annotate(wrongname=[sg1])
        doc.annotate(alsowrongname=[sg2])
        r_dict = {'key': 'abc-123', 'value': 'xyz-999'}
        with self.assertRaises(AttributeError):
            Relation.from_json(relation_dict=r_dict, doc=doc)

        # relations can have metadata too
        sg1 = SpanGroup(spans=[], id=123, field='abc')
        sg2 = SpanGroup(spans=[], id=999, field='xyz')
        r = Relation(key=sg1, value=sg2, id=40404,
                     doc=Document(symbols='doesnt-get-used-when-to-json'), field='same-here',
                     metadata=Metadata(foo='bar'))
        r_dict = {'key': 'abc-123', 'value': 'xyz-999', 'id': 40404, 'metadata': {'foo': 'bar'}}
        assert r.to_json() == r_dict
        doc = Document(symbols='a b c d e f g')
        sg1 = SpanGroup(spans=[], id=123)
        sg2 = SpanGroup(spans=[], id=999)
        doc.annotate(abc=[sg1])
        doc.annotate(xyz=[sg2])
        assert Relation.from_json(relation_dict=r_dict, doc=doc).to_json() == r_dict
