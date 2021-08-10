from mmda.types.document import Document
from typing import Union, List, Dict, Any

from mmda.types.document import Document
from mmda.models.base_predictors.base_predictor import BasePredictor

class BaseHFTransformersModelPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["transformers", "torch"]

    def __init__(self, model: Any, config: Any, tokenizer: Any):

        super().__init__(model, config)
        self.tokenizer = tokenizer