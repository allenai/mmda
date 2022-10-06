import unittest

from pydantic.error_wrappers import ValidationError

import ai2_internal.api as mmda_api
import mmda.types.annotation as mmda_ann
from mmda.types import Metadata
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
            'metadata': {'text': 'hello', 'id': 1}
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

        # we need to manually set the uuids to be equal
        # because by default they are randomly generated
        sg_ann.uuid = sg_ann_2.uuid = 'manually-fix-to-avoid-randomness'

        self.assertEqual(sg_ann, sg_ann_2)

    def test_box(self):
        box = mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0)
        assert box.to_mmda() == mmdaBox(l=0.1, t=0.1, w=0.1, h=0.1, page=0)
        assert mmda_api.Box.from_mmda(box.to_mmda()) == box

    def test_span(self):
        span = mmda_api.Span(start=0, end=1, box=mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0))
        assert span.to_mmda() == mmdaSpan(start=0, end=1, box=mmdaBox(l=0.1, t=0.1, w=0.1, h=0.1, page=0))

    def test_box_group(self):
        box_group = mmda_api.BoxGroup(boxes=[mmda_api.Box(left=0.1, top=0.1, width=0.1, height=0.1, page=0)],
                                      id=0, type='test', attributes={'one': 'Test string'})

        assert box_group.to_mmda() == []


    def test_span_group(self):
        span_group = mmda_api.SpanGroup(spans=[], box_group=mmda_api.BoxGroup(boxes=[mmda_api.Box(left=0.1, top=0.1,
                                                                                                 width=0.1, height=0.1,
                                                                                                 page=0)],
                                       id=0, type='test', attributes={'one': 'Test string'}),
                                       attributes={'one': 'Test string'}
                                       )

        assert span_group.to_mmda() == []