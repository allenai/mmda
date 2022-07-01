"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings, Field

from mmda.predictors.hf_predictors.token_classification_predictor import (
    IVILATokenClassificationPredictor,
)
from mmda.types import api
from mmda.types.document import Document, SpanGroup
from mmda.types.image import frombase64


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.
    """

    symbols: str
    images: List[str]
    tokens: List[api.SpanGroup]
    rows: List[api.SpanGroup]
    pages: List[api.SpanGroup]
    blocks: List[api.BoxGroup]

    def to_mmda(self):
        doc = Document(symbols=self.symbols)
        doc.annotate(tokens=[sg.to_mmda() for sg in self.tokens])
        doc.annotate(rows=[sg.to_mmda() for sg in self.rows])
        doc.annotate(pages=[sg.to_mmda() for sg in self.pages])
        doc.annotate(blocks=[bg.to_mmda() for bg in self.blocks])

        images = [frombase64(img) for img in self.images]
        doc.annotate_images(images)
        return doc


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """

    groups: List[api.SpanGroup]

    @classmethod
    def from_mmda(cls, groups: List[SpanGroup]) -> "Prediction":
        return cls(groups=[api.SpanGroup.from_mmda(grp) for grp in groups])


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.
    """

    subpage_per_run: int = Field(
        default=2,
        description="The maximum number of subpages we can send to the models at one time. "
                    "Used for capping the maximum memory usage during the vila dep."
    )


class Predictor:
    """
    Interface on to underlying VILA Predictor.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        self._predictor = IVILATokenClassificationPredictor.from_pretrained(
            self._artifacts_dir,
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.
        """
        predictions = []

        for inst in instances:
            span_groups = self._predictor.predict(
                inst.to_mmda(), subpage_per_run=self._config.subpage_per_run
            )
            predictions.append(
                Prediction(groups=[api.SpanGroup.from_mmda(sg) for sg in span_groups])
            )

        return predictions
