"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

import logging
from typing import List

import torch
from pydantic import BaseModel, BaseSettings, Field

from ai2_internal.api import BoxGroup
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.types import image
from mmda.types.document import Document

logger = logging.getLogger(__name__)


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    Input is a list of page images, base64-encoded"""

    page_images: List[str] = Field(description="List of base64-encoded page images")

class Prediction(BaseModel):
    """Output is a set of bounding boxes with metadata"""

    groups: List[BoxGroup] = Field(description="PDF Text Regions")


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.
    """
    weights_paths = ["lp://efficientdet/PubLayNet", "lp://efficientdet/MFD"]

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
        Performs the start-up operations required to
        ready the model for inference.

        LayoutPraser uses pre-trained PubLayNet and MFD models managed
        by the underlying layoutparser tool:
        https://layout-parser.readthedocs.io/en/latest/api_doc/models.html
        """
        device = "cuda" if torch.cuda.is_available() else None

        if device == "cuda":
            logger.info("CUDA device detected, running model with GPU acceleration.")
        else:
            logger.info("No CUDA device detected, running model on CPU.")

        self._lp_predictors = [
            LayoutParserPredictor.from_pretrained(weights_path, device=device)
            for weights_path in self._config.weights_paths
        ]

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        images = [image.frombase64(im) for im in instance.page_images]
        doc = Document(symbols="")
        doc.annotate_images(images)

        box_groups = []
        for predictor in self._lp_predictors:
            box_groups.extend(predictor.predict(doc))

        return Prediction(groups=[BoxGroup.from_mmda(bg) for bg in box_groups])

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
