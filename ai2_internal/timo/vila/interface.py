"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings, Field

from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
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
    def from_mmda(cls, groups: List[SpanGroup]) -> 'Prediction':
        return cls(groups=[api.SpanGroup.from_mmda(grp) for grp in groups])


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.
    """

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

    @property
    def label_names(self):
        return {int(key): val for key, val in self.DOCBANK_LABEL_MAP.items()}


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
        self._predictor = IVILAPredictor.from_pretrained(
            self._artifacts_dir,
            added_special_sepration_token="[BLK]",
            agg_level="row",
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.
        """
        predictions = []

        for inst in instances:
            span_groups = self._predictor.predict(inst.to_mmda())
            for span_group in span_groups:
                span_group.type = self._config.label_names.get(
                    span_group.type, f"label{span_group.type}"
                )
            predictions.append(
                Prediction(groups=[api.SpanGroup.from_mmda(sg) for sg in span_groups])
            )

        return predictions
