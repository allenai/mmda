from abc import abstractmethod
from typing import Union, List, Dict, Any

from mmda.types.document import Document
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class BaseSpacyPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["spacy"]


# For sentence boundary detector, please check
# https://github.com/allenai/VILA/blob/e1a7ad51e5414bf202ee273285d24a6993a3f1b6/src/vila/dataset/preprocessors/layout_indicator.py#L14