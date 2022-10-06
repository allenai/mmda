import unittest

from pydantic.error_wrappers import ValidationError

import ai2_internal.api as mmda_api
import mmda.types.annotation as mmda_ann


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
