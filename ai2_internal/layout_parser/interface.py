"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.types import image
from mmda.types.api import BoxGroup
from mmda.types.document import Document

from pydantic import BaseModel, BaseSettings, Field


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """
    """Input is a list of page images, base64-encoded"""

    page_images: List[str] = Field(description="List of base64-encoded page images")


    # field1: str = Field(description="Some field of consequence")
    # field2: float = Field(description="Some other field of consequence")


class Prediction(BaseModel):
    """Output is a set of bounding boxes with metadata"""

    groups: List[BoxGroup] = Field(description="PDF Text Regions")


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
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
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self._lp_predictors = [
            LayoutParserPredictor.from_pretrained(weights_path)
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
