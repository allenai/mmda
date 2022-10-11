import unittest

from pydantic.error_wrappers import ValidationError

import ai2_internal.api as mmda_api
import mmda.types.annotation as mmda_ann
from mmda.types.document import Document
from mmda.types.box import Box as mmdaBox
from mmda.types.span import Span as mmdaSpan


class ClassificationAttributes(mmda_api.Attributes):
    label: str
    score: float


class ClassificationSpanGroup(mmda_api.SpanGroup):
    attributes: ClassificationAttributes


class TestApi(unittest.TestCase):
    def test_vanilla_span_group(self) -> None:
        sg_ann = mmda_ann.SpanGroup.from_json({
            'spans': [{'start': 0, 'end': 1}],
            'id': 1,
            'metadata': {'text': 'hello', 'id': 999}    # note id not used; it's just in metadata
        })

        sg_api = mmda_api.SpanGroup.from_mmda(sg_ann)

        self.assertEqual(sg_api.text, 'hello')
        self.assertEqual(sg_api.id, 1)
        self.assertEqual(sg_api.attributes.dict(), {})

    def test_classification_span_group(self) -> None:
        sg_ann = mmda_ann.SpanGroup.from_json({
            'spans': [{'start': 0, 'end': 1}],
            'metadata': {'text': 'hello', 'id': 1}
        })

        with self.assertRaises(ValidationError):
            # this should fail because metadata is missing label
            # and confidence
            ClassificationSpanGroup.from_mmda(sg_ann)

        sg_ann.metadata.label = 'label'
        sg_ann.metadata.score = 0.5

        sg_api = ClassificationSpanGroup.from_mmda(sg_ann)
        self.assertEqual(
            sg_api.attributes.dict(), {'label': 'label', 'score': 0.5}
        )

        # extra field should just get ignored
        sg_ann.metadata.extra = 'extra'
        self.assertEqual(
            sg_api.attributes.dict(), {'label': 'label', 'score': 0.5}
        )

        with self.assertRaises(ValidationError):
            # this should fail bc score is not a float
            sg_ann.metadata.score = 'not a float'
            ClassificationSpanGroup.from_mmda(sg_ann)

    def test_equivalence(self):
        sg_ann = mmda_ann.SpanGroup.from_json({
            'spans': [{'start': 0, 'end': 1}],
            'metadata': {'label': 'label', 'score': 0.5}
        })
        sg_ann_2 = ClassificationSpanGroup.from_mmda(sg_ann).to_mmda()
        self.assertDictEqual(sg_ann.to_json(), sg_ann_2.to_json())
        self.assertDictEqual(sg_ann.__dict__, sg_ann_2.__dict__)


    def test_box(self):
        box = mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0)
        assert box.to_mmda() == mmdaBox(l=0.1, t=0.1, w=0.1, h=0.1, page=0)
        assert mmda_api.Box.from_mmda(box.to_mmda()) == box

    def test_span(self):
        span = mmda_api.Span(start=0, end=1, box=mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0))
        assert span.to_mmda() == mmdaSpan(start=0, end=1, box=mmdaBox(l=0.1, t=0.1, w=0.1, h=0.1, page=0))

    def test_box_group(self):
        box_group = mmda_api.BoxGroup(
            boxes=[
                mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0)
            ],
            id=0,
            type='test',
            # these attributes are going to be discarded because
            # BoxGroup is using the default Attributes class
            attributes={'one': 'Test string'}
        )

        self.assertEqual(
            mmda_api.BoxGroup.from_mmda(box_group.to_mmda()),
            box_group
        )

    def test_span_group(self):
        box_group = mmda_api.BoxGroup(
            boxes=[
                mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0)
            ],
            id=0,
            type='test',
            attributes={'one': 'Test string'}
        )

        span_group = mmda_api.SpanGroup(
            spans=[],
            box_group=box_group,
            attributes={'one': 'Test string'},
            id=0,
            type='test',
            text='this is a test'
        )

        self.assertEqual(
            mmda_api.SpanGroup.from_mmda(span_group.to_mmda()),
            span_group
        )

    def test_doc_to_mmda(self):
        doc = {
            'symbols': 'LinkBERT: Language Model Pretraining with Document Link Knowledge\nMichihiro Yasunaga 1 Jure Leskovec',
            'tokens': [{'spans': [{'start': 0,
                         'end': 9,
                         'box': {'left': 0.11994771241830066,
                                 'top': 0.11309994419191913,
                                 'width': 0.11718407483660132,
                                 'height': 0.018113888888888974,
                                 'page': 0}}],
              'box_group': None,
              'id': 0,
              'type': None,
              'text': None},
             {'spans': [{'start': 10,
                         'end': 18,
                         'box': {'left': 0.24182008790849677,
                                 'top': 0.11309994419191913,
                                 'width': 0.09899346830065353,
                                 'height': 0.018113888888888974,
                                 'page': 0}}],
              'box_group': None,
              'id': 1,
              'type': None,
              'text': None}],
            'rows': [{'spans': [{'start': 0,
    'end': 65,
    'box': {'left': 0.11994771241830066,
     'top': 0.11309994419191913,
     'width': 0.7354068405228755,
     'height': 0.018113888888888985,
     'page': 0}}],
     'box_group': None,
     'id': 0,
     'type': None,
     'text': None},
     {'spans': [{'start': 66,
        'end': 120,
        'box': {'left': 0.2959248366013072,
         'top': 0.1787468949494949,
         'width': 0.3815858895424835,
         'height': 0.014096770959595939,
         'page': 0}}],
      'box_group': None,
      'id': 1,
      'type': None,
      'text': None}],
                'pages':
     [{'spans': [{'start': 0,
        'end': 6499,
        'box': {'left': 0.09018137254901962,
         'top': 0.0,
         'width': 0.9098186274509804,
         'height': 0.9055718707070707,
         'page': 0}}],
      'box_group': None,
      'id': 0,
      'type': None,
      'text': None},
     {'spans': [{'start': 6500,
        'end': 14963,
        'box': {'left': 0.08976797385620916,
         'top': 0.059364365656565665,
         'width': 0.7977906869836605,
         'height': 0.8429158383838383,
         'page': 1}}],
      'box_group': None,
      'id': 1,
      'type': None,
      'text': None}],
                'blocks': [{'boxes': [{'left': 0.5080462187723397,
        'top': 0.5965596593991674,
        'width': 0.38663263258591196,
        'height': 0.24100791083441842,
        'page': 0}],
      'id': None,
      'type': 'Text'},
        {'boxes': [{'left': 0.49845482321346507,
           'top': 0.4326159544665404,
           'width': 0.38783658096213747,
           'height': 0.13613448480162957,
           'page': 0}],
         'id': None,
         'type': 'Text'}],
                'images': [
                    'iVBORw0KGgoAAAANSUhEUgAAAmQAAAMYCAIAAADq5GzlAAEAAElEQVR4nOyddXRURxfA7+5mNxt39xAhCW7BQoIGdylFUrSlFFq8',
                    'iVBORw0KGgoAAAANSUhEUgAAAmQAAAMYCAIAAADq5GzlAAEAAElEQVR4nOydd3gUxf/HZ/dq7tJ7D2mkEDoBQu9FuoA0FRARFVR+',
                           ]
            }

        mmda_doc = mmda_api.doc_to_mmda(doc)
        assert isinstance(mmda_doc, Document)
        assert isinstance(mmda_doc.symbols, str)
        assert isinstance(mmda_doc.tokens, list)
        assert isinstance(mmda_doc.tokens[0], mmda_ann.SpanGroup)
        assert isinstance(mmda_doc.rows[0], mmda_ann.SpanGroup)
        assert isinstance(mmda_doc.pages[0], mmda_ann.SpanGroup)
        assert isinstance(mmda_doc.blocks[0], mmda_ann.SpanGroup)
