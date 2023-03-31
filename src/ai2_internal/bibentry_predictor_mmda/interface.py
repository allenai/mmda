"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings

from ai2_internal import api
from mmda.predictors.hf_predictors.bibentry_predictor.predictor import BibEntryPredictor
from mmda.predictors.hf_predictors.bibentry_predictor.types import BibEntryStructureSpanGroups
from mmda.types.document import Document


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
    bib_entry_boxes: List[api.BoxGroup]

    def to_mmda(self) -> Document:
        doc = Document(symbols=self.symbols)
        doc.annotate(tokens=[token.to_mmda() for token in self.tokens])
        doc.annotate(pages=[page.to_mmda() for page in self.pages])
        doc.annotate(bib_entry_boxes=[bbox.to_mmda() for bbox in self.bib_entry_boxes])

        return doc


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """
    bib_entry_number: List[api.SpanGroup]
    bib_entry_authors: List[api.SpanGroup]
    bib_entry_title: List[api.SpanGroup]
    bib_entry_venue_or_event: List[api.SpanGroup]
    bib_entry_year: List[api.SpanGroup]
    bib_entry_doi: List[api.SpanGroup]
    bib_entry_url: List[api.SpanGroup]

    @staticmethod
    def from_mmda(mmda: BibEntryStructureSpanGroups) -> 'Prediction':
        return Prediction(
            bib_entry_number=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_number],
            bib_entry_authors=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_authors],
            bib_entry_title=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_title],
            bib_entry_venue_or_event=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_venue_or_event],
            bib_entry_year=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_year],
            bib_entry_doi=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_doi],
            bib_entry_url=[api.SpanGroup.from_mmda(sg) for sg in mmda.bib_entry_url],
        )


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
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
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self._predictor = BibEntryPredictor(self._artifacts_dir)

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
        documents = [i.to_mmda() for i in instances]
        mmda_predictions = [self._predictor.predict(document) for document in documents]
        return [Prediction.from_mmda(prediction) for prediction in mmda_predictions]
