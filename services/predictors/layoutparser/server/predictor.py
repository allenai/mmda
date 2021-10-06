import json
import os
from typing import List

from pydantic import BaseModel, BaseSettings, Field

from server.api_model import Instance, BoxGroup
import mmda.predictors.lp_predictors as lpp
from mmda.types.document import Document
from mmda.types.image import Image


class PredictorConfig(BaseSettings):
    """
    Path to the model weights
    """

    MODEL_CONFIG = Field("Comma-separated list of lp://... config paths")


class Predictor:
    """
    Initialize model
    """

    _lp_predictor: lpp.LayoutParserPredictor

    def __init__(self, config: PredictorConfig):
        """
        Load weights
        """
        self.model_paths = config.MODEL_CONFIG.split(",")
        self._lp_predictors = [
            lpp.LayoutParserPredictor.from_pretrained(path) for path in self.model_paths
        ]

    def predict_batch(self, instances: List[Instance]) -> List[List[BoxGroup]]:
        predictions = []

        for instance in instances:
            doc = Document(
                symbols="", images=[Image.frombase64(i) for i in instance.images]
            )
            doc_predictions = []
            for i, pred in enumerate(self._lp_predictors):
                annotations = pred.predict(doc)
                doc_predictions.extend(annotations)
            predictions.append(doc_predictions)

        return predictions
