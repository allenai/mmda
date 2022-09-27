"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, Field

from mmda.predictors.heuristic_predictors.figure_table_predictors import ObjectCaptionMap, FigureTablePredictions
from mmda.types import image
from mmda.types.document import Document
from mmda.types.image import frombase64

from ai2_internal import api


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    symbols: str
    tokens: List[api.SpanGroup]
    rows: List[api.SpanGroup]
    pages: List[api.SpanGroup]
    vila_span_groups: List[api.SpanGroup]
    layoutparser_span_groups: List[api.SpanGroup]
    page_images: List[str] = Field(description="List of base64-encoded page images")

    def to_mmda(self):
        doc = Document(symbols=self.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in self.tokens])
        doc.annotate(rows=[sg.to_mmda() for sg in self.rows])
        doc.annotate(pages=[sg.to_mmda() for sg in self.pages])
        doc.annotate(blocks=[bg.to_mmda() for bg in self.blocks])

        images = [frombase64(img) for img in self.images]
        doc.annotate_images(images)
        doc.annotate(vila_span_groups=[sg.to_mmda for sg in self.vila_span_groups])
        doc.annotate(layoutparser_span_groups=[sg.to_mmda for sg in self.layoutparser_span_groups])
        return doc


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    table_figure_caption_list: List[ObjectCaptionMap]


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

    def __init__(self, dpi: int = 72):
        self.dpi = dpi
        self._predictor = FigureTablePredictions(self.dpi)

    def predict_one(self, inst: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        doc = Document(symbols=inst.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in inst.tokens])
        doc.annotate(rows=[sg.to_mmda() for sg in inst.rows])
        doc.annotate(pages=[sg.to_mmda() for sg in inst.pages])
        images = [image.frombase64(im) for im in inst.page_images]
        doc.annotate_images(images)
        doc.annotate(vila_span_groups=[sg.to_mmda() for sg in inst.vila_span_groups])
        doc.annotate(layoutparser_span_groups=[sg.to_mmda() for sg in inst.vila_span_groups])

        predictions_table_figure_list = self._predictor.predict(inst.to_mmda())

        prediction = Prediction(
            table_figure_caption_map=predictions_table_figure_list)

        return prediction

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
