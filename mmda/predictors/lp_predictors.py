from abc import abstractmethod
from typing import Union, List, Dict, Any

from mmda.types.names import *
from mmda.types.document import Document
from mmda.types.annotation import BoxGroup
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class LayoutParserPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images]

    def __init__(self, model: Any, config: Any):

        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        pass