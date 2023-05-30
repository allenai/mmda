"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings, Field

from ai2_internal.api import SpanGroup
from mmda.predictors.heuristic_predictors.sentence_boundary_predictor import (
    PysbdSentenceBoundaryPredictor,
)
from mmda.types.document import Document


class Instance(BaseModel):
    """
    Describes one Instance from which the model attempts to extract sentences.
    """

    symbols: str = Field(description="Full text of the document")
    tokens: List[SpanGroup] = Field(description="All the tokens in the document")
    pages: List[SpanGroup] = Field(description="The pages of the document")


class Prediction(BaseModel):
    """
    The output prediction for one Instance
    """

    sentences: List[SpanGroup] = Field(description="Extracted sentences")


class PredictorConfig(BaseSettings):
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
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self._model = PysbdSentenceBoundaryPredictor()

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Predicts sentences for one document.
        """
        doc = Document(instance.symbols)
        doc.annotate(
            tokens=[sg.to_mmda() for sg in instance.tokens],
            pages=[sg.to_mmda() for sg in instance.pages]
        )
        sentences = self._model.predict(doc)

        return Prediction(
            sentences=[SpanGroup.from_mmda(sentence) for sentence in sentences]
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.
        """
        return [self.predict_one(instance) for instance in instances]
