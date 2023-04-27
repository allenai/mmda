"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List
import time
import logging

from pydantic import BaseModel, Field, BaseSettings

from mmda.predictors.heuristic_predictors.figure_table_predictors import FigureTablePredictions
from mmda.types.document import Document
from ai2_internal import api

logger = logging.getLogger(__name__)

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
    pages: List[api.SpanGroup]
    vila_span_groups: List[api.SpanGroup]
    blocks: List[api.BoxGroup]

    def to_mmda(self):
        start = time.time()
        doc = Document(symbols=self.symbols)
        logger.debug(f"Instantiating document with symbols {(time.time() - start):.3f} seconds")
        start = time.time()
        doc.annotate(tokens=[sg.to_mmda() for sg in self.tokens])
        logger.debug(f"Tokens annotation {(time.time() - start):.3f} seconds")
        start = time.time()
        doc.annotate(pages=[sg.to_mmda() for sg in self.pages])
        logger.debug(f"Pages annotation {(time.time() - start):.3f} seconds")
        start = time.time()
        doc.annotate(blocks=[sg.to_mmda() for sg in self.blocks])
        logger.debug(f"Blocks annotation {(time.time() - start):.3f} seconds")
        start = time.time()
        doc.annotate(vila_span_groups=[sg.to_mmda() for sg in self.vila_span_groups])
        logger.debug(f"Vila span_groups annotation {(time.time() - start):.3f} seconds")
        return doc


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    figures: List[api.BoxGroup]
    figure_captions: List[api.SpanGroup]
    figure_to_figure_captions: List[api.Relation]
    tables: List[api.BoxGroup]
    table_captions: List[api.SpanGroup]
    table_to_table_captions: List[api.Relation]


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

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        predictions_table_figure_dict = FigureTablePredictions(instance.to_mmda()).predict()
        return Prediction(
            figures=[api.BoxGroup.from_mmda(sg) for sg in predictions_table_figure_dict['figures']],
            figure_captions=[api.SpanGroup.from_mmda(bg) for bg in predictions_table_figure_dict['figure_captions']],
            figure_to_figure_captions=predictions_table_figure_dict['figure_to_figure_captions'],
            tables=[api.BoxGroup.from_mmda(sg) for sg in predictions_table_figure_dict['tables']],
            table_captions=[api.SpanGroup.from_mmda(bg) for bg in predictions_table_figure_dict['table_captions']],
            table_to_table_captions=predictions_table_figure_dict['table_to_table_captions']
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
