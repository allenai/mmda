from abc import abstractmethod
from typing import Union, List, Dict, Any

from mmda.types.names import *
from mmda.types.document import Document
from mmda.types.annotation import BoxGroup
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class BaseLayoutParserPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images]

    def __init__(self, model: Any, config: Any):

        self.model = model
        self.config = config

