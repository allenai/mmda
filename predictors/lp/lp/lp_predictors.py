from typing import Union, List, Dict, Any

import layoutparser as lp

from mmda.types.names import *
from mmda.types.document import Document
from mmda.types.box import Box
from mmda.predictors.base_predictor import BasePredictor


class LayoutParserPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images]

    def __init__(self, model: lp.Detectron2LayoutModel):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        model_path: str = None,
        label_map: Dict = None,
        extra_config: List = None,
        enforce_cpu: bool = False,
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

        model = lp.Detectron2LayoutModel(
            config_path,
            model_path=model_path,
            label_map=label_map,
            extra_config=extra_config,
            enforce_cpu=enforce_cpu,
        )

        return cls(model)

    def postprocess(self, model_outputs: lp.Layout, page_index: int) -> List[Box]:
        """Convert the model outputs into the Annotation format"""

        # block.coordinates returns the left, top, bottom, right coordinates
        return [
            Box(
                l=block.coordinates[0],
                t=block.coordinates[1],
                w=block.width,
                h=block.height,
                page=page_index,
            )
            for block in model_outputs
        ]

    def predict(self, document: Document) -> List[Box]:

        document_prediction = []

        for image_index, image in enumerate(document.images):
            model_outputs = self.model.detect(image)
            document_prediction.extend(self.postprocess(model_outputs, image_index))

        return document_prediction