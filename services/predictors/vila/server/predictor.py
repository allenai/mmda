from typing import List

from pydantic import BaseSettings

from mmda.predictors.hf_predictors.vila_predictor import HVILAPredictor
from mmda.types.document import Document
from mmda.types.annotation import SpanGroup


class PredictorConfig(BaseSettings):
    """Uninitialized fields must be set via environment variables"""

    ARTIFACTS_DIR: str
    DOCBANK_LABEL_MAP = {
        "0": "Title",
        "1": "Author",
        "2": "Abstract",
        "3": "Keywords",
        "4": "Section",
        "5": "Paragraph",
        "6": "List",
        "7": "Bibliography",
        "8": "Equation",
        "9": "Algorithm",
        "10": "Figure",
        "11": "Table",
        "12": "Caption",
        "13": "Header",
        "14": "Footer",
        "15": "Footnote",
    }
    label_names = {int(key): val for key, val in DOCBANK_LABEL_MAP.items()}


class Predictor:
    """
    Initialize simple VILA Predictor
    """

    _config: PredictorConfig

    def __init__(self, config: PredictorConfig):
        """
        Initialize your model using the passed parameters
        """
        self._config = config
        self._load_model()

    def _load_model(self):
        self._predictor = HVILAPredictor.from_pretrained(
            self._config.ARTIFACTS_DIR, agg_level="block", group_bbox_agg="first"
        )

    def predict_batch(self, instances: List[Document]) -> List[List[SpanGroup]]:
        predictions = []

        for doc in instances:
            annotations = self._predictor.predict(doc)
            for a in annotations:
                a.type = self._config.label_names.get(a.type, f"label{a.type}")
            predictions.append(annotations)

        return predictions
