"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings

from ai2_internal import api
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.
    """
    symbols: str
    tokens: List[api.SpanGroup]
    pages: List[api.SpanGroup]


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    mentions: List[api.SpanGroup]


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.
    """
    pass


def group_by_line(boxes):
    boxes = sorted(boxes, key=lambda box: box.t)
    return [list(line_boxes) for t, line_boxes in groupby(boxes, key=lambda box: box.t)]

def calc_bounding_box(boxes):
    l = min(b.l for b in boxes)
    t = min(b.t for b in boxes)
    w = max(b.l + b.w for b in boxes) - l
    h = max(b.t + b.h for b in boxes) - t
    return mmda.types.box.Box(l=l, t=t, w=w, h=h, page=boxes[0].page)

def merge_boxes(boxes):
    boxes_by_line = group_by_line(boxes)
    return [calc_bounding_box(line_boxes) for line_boxes in boxes_by_line]

def merge_boxes_of_sg(sg):
    sg.box_group.boxes = merge_boxes(sg.box_group.boxes)

def all_spans_close(sg):
    spans = sorted(sg.spans, key=lambda span: span.start)
    return all(span.end <= next_span.start <= span.end + 5 for span, next_span in zip(spans, spans[1:]))

def merge_citation_mention_boxes(citation_mentions):
    for sg in citation_mentions:
        if all_spans_close(sg):
            merge_boxes_of_sg(sg)


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """
    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._predictor = MentionPredictor(artifacts_dir)

    def predict_one(self, inst: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        doc = Document(symbols=inst.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in inst.tokens])
        doc.annotate(pages=[sg.to_mmda() for sg in inst.pages])

        prediction_span_groups = self._predictor.predict(doc)
        merge_citation_mention_boxes(prediction_span_groups)
        doc.annotate(citation_mentions=prediction_span_groups)


        return Prediction(mentions=[api.SpanGroup.from_mmda(sg) for sg in doc.citation_mentions])

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        return [self.predict_one(instance) for instance in instances]
