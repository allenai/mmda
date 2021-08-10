from abc import abstractmethod
from mmda.types.annotation import Annotation
from typing import Union, List, Dict, Any

from mmda.types.document import Document
from mmda.types.annotation import Annotation
from mmda.models.base_predictors.base_predictor import BasePredictor


class BaseHeuristicPredictor(BasePredictor):
    REQUIRED_BACKENDS = []