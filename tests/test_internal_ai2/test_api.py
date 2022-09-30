import unittest

from pydantic.error_wrappers import ValidationError

import ai2_internal.api as mmda_api
import mmda.types.annotation as mmda_ann


class ClassificationMetadata(mmda_api.Metadata):
    label: str
    score: float


class ClassificationSpanGroup(mmda_api.SpanGroup):
    metadata: ClassificationMetadata


class TestApi(unittest.TestCase):
    def test_vanilla_span_group(self) -> None:
        sg_ann = mmda_ann.SpanGroup.from_json({
            'spans': [{'start': 0, 'end': 1}],
            'metadata': {'text': 'hello', 'id': 1}
        })

        sg_api = mmda_api.SpanGroup.from_mmda(sg_ann)

        self.assertEqual(sg_api.text, 'hello')
        self.assertEqual(sg_api.id, 1)
        self.assertEqual(sg_api.metadata.dict(), {})

    def test_classification_span_group(self) -> None:
        sg_ann = mmda_ann.SpanGroup.from_json({
            'spans': [{'start': 0, 'end': 1}],
            'metadata': {'text': 'hello', 'id': 1}
        })

        with self.assertRaises(ValidationError):
            # this should fail because metadata is missing label and confidence
            ClassificationSpanGroup.from_mmda(sg_ann)

        sg_ann.metadata.label = 'label'
        sg_ann.metadata.score = 0.5

        sg_api = ClassificationSpanGroup.from_mmda(sg_ann)
        self.assertEqual(
            sg_api.metadata.dict(), {'label': 'label', 'score': 0.5}
        )

        # extra field should just get ignored
        sg_ann.metadata.extra = 'extra'
        self.assertEqual(
            sg_api.metadata.dict(), {'label': 'label', 'score': 0.5}
        )

        with self.assertRaises(ValidationError):
            # this should fail bc score is not a float
            sg_ann.metadata.score = 'not a float'
            ClassificationSpanGroup.from_mmda(sg_ann)
