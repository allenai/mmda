from typing import Union, List, Dict, Any, Optional

from tqdm import tqdm
import layoutparser as lp

from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
# from mmda.types.names import Pages, Images, Tokens # TODO determine
from mmda.types.names import Pages, Images
from mmda.types.span import Span
from mmda.predictors.base_predictors.base_predictor import BasePredictor


class BibEntryDetectionPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser", "detectron2"]
    # REQUIRED_DOCUMENT_FIELDS = [Pages, Images, Tokens] # TODO determine
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images]

    def __init__(self, artifacts_dir: str, threshold: float = 0.90):
        label_map = {0: "bibentry"}
        self.model = lp.Detectron2LayoutModel(
            config_path=f"{artifacts_dir}/config.yaml",
            model_path=f"{artifacts_dir}/model_final.pth",
            label_map=label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", threshold]
        )

    def postprocess(self,
                    model_outputs: lp.Layout,
                    # TODO i am confused about tokens...
                    # tokens: List[SpanGroup],
                    tokens: List[List[Span]],
                    page_index: int,
                    image: "PIL.Image") -> List[SpanGroup]:
        """Convert the model outputs for a single page image into the mmda format

        Args:
            model_outputs (lp.Layout):
                The layout detection results from the layoutparser model for
                a page image
            tokens (List[SpanGroup]):
                The document's tokens for this page
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
            SpanGroup(
                spans=[
                    Span(
                        # TODO: these start/ends are page token IDs?
                        start=,
                        end=,
                        # TODO: use coords of box tightened around the tokens
                        box=Box.from_coordinates(
                            x1=ele.block.x_1,
                            y1=ele.block.y_1,
                            x2=ele.block.x_2,
                            y2=ele.block.y_2,
                            page=page_index
                        ).get_relative(
                            page_width=page_width,
                            page_height=page_height,
                        )
                    )
                ],
                type=ele.type,
            )
            for ele in model_outputs
        ]

    def predict(self, document: Document) -> List[SpanGroup]:
        """Returns a list of SpanGroups for the detected layouts for all pages

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
                # TODO tokens .. ?
                self.postprocess(model_outputs, document.tokens[image_index], image_index, image)
            )

        return document_prediction
