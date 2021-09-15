from typing import Union, List, Dict, Any, Optional

from tqdm import tqdm
import layoutparser as lp

from mmda.types.names import *
from mmda.types.document import Document
from mmda.types.box import Box
from mmda.types.annotation import BoxGroup, Annotation
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class LayoutParserPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images]

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        config_path: str = 'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
        model_path: str = None,
        label_map: Optional[Dict] = None,
        extra_config: Optional[Dict] = None,
        device: str = None,
    ):
        """Initialize a pre-trained layout detection model from
        layoutparser. The parameters currently are the same as the
        default layoutparser Detectron2 models
        https://layout-parser.readthedocs.io/en/latest/api_doc/models.html ,
        and will be updated in the future.
        """

        # TODO[shannon]: In the current version of layoutparser (v0.2.0),
        # only Detectron2 models loaded. And we will modify it in the future
        # such that we can load models using different DL backends.

        model = lp.AutoLayoutModel(
            config_path = config_path,
            model_path = model_path,
            label_map = label_map,
            extra_config = extra_config,
            device = device,
        )

        return cls(model)

    def postprocess(self, model_outputs: lp.Layout, page_index: int) -> List[BoxGroup]:
        """Convert the model outputs into the Annotation format"""

        # block.coordinates returns the left, top, bottom, right coordinates
        return [
            BoxGroup(boxes = [Box(
                l=block.coordinates[0],
                t=block.coordinates[1],
                w=block.width,
                h=block.height,
                page=page_index,
            )], type = block.type)
            for block in model_outputs
        ]

    def predict(self, document: Document) -> List[Annotation]:

        document_prediction = []

        for image_index, image in enumerate(tqdm(document.images)):
            model_outputs = self.model.detect(image)
            document_prediction.extend(self.postprocess(model_outputs, image_index))

        return document_prediction