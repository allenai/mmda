from typing import List

from pydantic import BaseModel, BaseSettings, Field

from ai2_internal.api import SpanGroup
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import (
    DictionaryWordPredictor
)
from mmda.types.document import Document


class Instance(BaseModel):
    """
    Inference input for a single paper.
    """
    symbols: str = Field(description="Extracted PDF document text")
    tokens: List[SpanGroup] = Field(description="The tokens to coerce into words")
    rows: List[SpanGroup] = Field(description="Document rows, used as signal for determining word boundaries")


class Prediction(BaseModel):
    """
    Inference output for a single paper.
    """
    words: List[SpanGroup] = Field(description="Input tokens coerced into words. Includes cleaned-up text.")


class PredictorConfig(BaseSettings):
    """
    no-op for this model
    """
    pass


class Predictor:
    """
    This class is instantiated at application startup as a singleton,
    and is used by the TIMO framework to interface with the underlying
    DWP predictor.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        self._predictor = DictionaryWordPredictor("/dev/null")

    def predict_one(self, instance: Instance) -> Prediction:
        doc = Document(instance.symbols)
        doc.annotate(tokens=[t.to_mmda() for t in instance.tokens])
        doc.annotate(rows=[r.to_mmda() for r in instance.rows])

        missing_row = 0
        missing_ids = []
        for token in doc.tokens:
            if len(token.rows) == 0:
                missing_row += 1
                missing_ids.append(token.id)

        print(f"Missing {missing_row} of {len(doc.tokens)}: {missing_ids}")

        words = self._predictor.predict(doc)

        # RE: https://github.com/allenai/scholar/issues/36200
        for word in words:
            if word.text:
                word.text = word.text.replace("\u0000", "")

        return Prediction(
            words=[SpanGroup.from_mmda(w) for w in words]
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.
        """
        return [self.predict_one(instance) for instance in instances]
