import itertools
from typing import Iterable

from mmda.predictors.hf_predictors.vila_predictor import (BaseVILAPredictor,
                                                          HVILAPredictor)
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.types.names import *
from mmda.types.nouveau.protocol import SpanPredictor


class NouveauHVilaPredictor(SpanPredictor):
    @classmethod
    def from_pretrained(cls, model_name: str, **preprocessor_config):
        return HVILAPredictor.from_pretrained(model_name, **preprocessor_config)

    def predict(self, document: Document) -> Iterable[SpanGroup]:
        preds = super().predict(document)

        import pdb
        pdb.set_trace()
