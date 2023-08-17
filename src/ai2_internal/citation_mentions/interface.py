"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List
from itertools import groupby
from bisect import bisect

from pydantic import BaseModel, BaseSettings

from ai2_internal import api
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor
from mmda.types.document import Document
from mmda.types.box import Box
from mmda.types.annotation import BoxGroup as MMDABoxGroup


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.
    """
    symbols: str
    tokens: List[api.SpanGroup]
    pages: List[api.SpanGroup]
    rows: List[api.SpanGroup]


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


def span_to_row_idx(doc, span):
    """
    assumes spans do not cross rows (meaning that span.start and
    span.end are assumed to be on the same row)
    """
    return bisect([r.start for r in doc.rows], span.start) - 1

def calc_bounding_box(boxes):
    l = min(b.l for b in boxes)
    t = min(b.t for b in boxes)
    w = max(b.l + b.w for b in boxes) - l
    h = max(b.t + b.h for b in boxes) - t
    return Box(l=l, t=t, w=w, h=h, page=boxes[0].page)

def spans_close(s1, s2):
    spans = sorted([s1, s2], key=lambda span: span.start)
    return all(span.end <= next_span.start <= span.end + 5 for span, next_span in zip(spans, spans[1:]))

def on_same_line(doc, s1, s2):
    return span_to_row_idx(doc, s1) == span_to_row_idx(doc, s2)

def build_box_group(sg):
    boxes = [span.box for span in sg.spans]
    return MMDABoxGroup(boxes=boxes)

def merge_span_group_boxes(doc, sg):
    boxes = []
    last_span, last_box = None, None
    for span, box in zip(sg.spans, sg.box_group.boxes):
        if last_span is not None and spans_close(last_span, span) and on_same_line(doc, last_span, span):
            last_box = calc_bounding_box([last_box, box])
            last_span = span
        else:
            if last_box is not None:
                boxes.append(last_box)
            last_span, last_box = span, box
    boxes.append(last_box)
    return boxes


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
        doc.annotate(rows=[sg.to_mmda() for sg in inst.rows])

        prediction_span_groups = self._predictor.predict(doc)
        box_groups = [build_box_group(sg) for sg in prediction_span_groups]
        # set box_groups and delete span boxes
        for sg, bg in zip(prediction_span_groups, box_groups):
            sg.box_group = bg
            for span in sg.spans:
                span.box = None
        for sg in prediction_span_groups:
            boxes = merge_span_group_boxes(doc, sg)
            sg.box_group.boxes = boxes
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
