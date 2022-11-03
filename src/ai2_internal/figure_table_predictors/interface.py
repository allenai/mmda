"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, Field, BaseSettings

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions
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
    images: List[str] = Field(description="List of base64-encoded page images")

    def to_mmda(self):
        doc = Document(symbols=self.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in self.tokens])
        doc.annotate(rows=[sg.to_mmda() for sg in self.rows])
        doc.annotate(pages=[sg.to_mmda() for sg in self.pages])
        doc.annotate_images([frombase64(img) for img in self.images])
        doc.annotate(layoutparser_span_groups=[sg.to_mmda() for sg in self.layoutparser_span_groups])
        doc.annotate(vila_span_groups=[sg.to_mmda() for sg in self.vila_span_groups])
        return doc


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    figure_list: List[api.SpanGroup]
    table_list: List[api.SpanGroup]


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.
    """
    pass


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
    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._predictor = FigureTablePredictions()

    def predict_one(self, inst: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        predictions_table_figure_list = self._predictor.predict(inst.to_mmda())
        return Prediction(
             figure_list=[api.SpanGroup.from_mmda(sg) for sg in predictions_table_figure_list[0]],
             table_list=[api.SpanGroup.from_mmda(sg) for sg in predictions_table_figure_list[1]]
        )

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
