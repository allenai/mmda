"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List, Tuple

from pydantic import BaseModel, BaseSettings

from ai2_internal import api
from mmda.predictors.xgb_predictors.citation_link_predictor import CitationLinkPredictor
from mmda.types.document import Document


# these should represent the extracted citation mentions and bibliography entries for a paper
class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.
    """
    symbols: str
    mentions: List[api.SpanGroup]
    bibs: List[api.SpanGroup]


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance
    """
    # tuple represents mention.id and bib.id for the linked pair
    linked_mentions: List[Tuple[str, str]]
    linked_mention_relations: List[api.Relation]


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
    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._predictor = CitationLinkPredictor(artifacts_dir)

    def predict_one(self, inst: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        doc = Document(symbols=inst.symbols)
        doc.annotate(mentions=[sg.to_mmda() for sg in inst.mentions])
        doc.annotate(bibs=[sg.to_mmda() for sg in inst.bibs])

        prediction = self._predictor.predict(doc) # returns [(mention.id, bib.id)]

        return Prediction(
            linked_mentions = prediction,
            linked_mention_relations = [
                api.Relation(from_id=mention_id, to_id=bib_id)
                for mention_id, bib_id in prediction
            ]
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
