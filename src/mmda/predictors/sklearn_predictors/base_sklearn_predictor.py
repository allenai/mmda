from abc import abstractmethod
from typing import Any, Dict, List, Union

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.document import Document


class BaseSklearnPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["sklearn", "numpy", "scipy", "tokenizers"]

    @classmethod
    def from_path(cls, tar_path: str):
        raise NotImplementedError
