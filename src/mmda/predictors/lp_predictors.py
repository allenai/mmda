from typing import Union, List, Dict, Any, Optional

from tqdm import tqdm
import layoutparser as lp

from mmda.types import Document, Box, BoxGroup, Metadata
from mmda.types.names import ImagesField, PagesField
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class LayoutParserPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser"]
    REQUIRED_DOCUMENT_FIELDS = [PagesField, ImagesField]

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        config_path: str = "lp://efficientdet/PubLayNet",
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

        model = lp.AutoLayoutModel(
            config_path=config_path,
            model_path=model_path,
            label_map=label_map,
            extra_config=extra_config,
            device=device,
        )

        return cls(model)

    def postprocess(self, 
        model_outputs: lp.Layout, 
        page_index: int,
        image: "PIL.Image") -> List[BoxGroup]:
        """Convert the model outputs into the mmda format

        Args:
            model_outputs (lp.Layout): 
                The layout detection results from layoutparser for 
                a page image
            page_index (int): 
                The index of the current page, used for creating the 
                `Box` object
            image (PIL.Image): 
                The image of the current page, used for converting
                to relative coordinates for the box objects

        Returns:
            List[BoxGroup]: 
            The detected layout stored in the BoxGroup format.
        """

        # block.coordinates returns the left, top, bottom, right coordinates

        page_width, page_height = image.size

        return [
            BoxGroup(
                boxes=[
                    Box(
                        l=block.coordinates[0],
                        t=block.coordinates[1],
                        w=block.width,
                        h=block.height,
                        page=page_index,
                    ).get_relative(
                        page_width=page_width,
                        page_height=page_height,
                    )
                ],
                metadata=Metadata(type=block.type)
            )
            for block in model_outputs
        ]

    def predict(self, document: Document) -> List[BoxGroup]:
        """Returns a list of Boxgroups for the detected layouts for all pages

        Args:
            document (Document): 
                The input document object 

        Returns:
            List[BoxGroup]: 
                The returned Boxgroups for the detected layouts for all pages
        """
        document_prediction = []

        for image_index, image in enumerate(tqdm(document.images)):
            model_outputs = self.model.detect(image)
            document_prediction.extend(
                self.postprocess(model_outputs, image_index, image)
            )

        return document_prediction