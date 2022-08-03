from typing import Union, List, Dict, Any, Optional

from tqdm import tqdm
import itertools
import layoutparser as lp

from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
# from mmda.types.names import Pages, Images, Tokens # TODO determine if Tokens needed in this line
from mmda.types.names import Pages, Images
from mmda.types.span import Span
from mmda.utils.tools import find_overlapping_tokens_for_box
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
                    page_tokens: List[Span],
                    page_index: int,
                    image: "PIL.Image") -> List[SpanGroup]:
        """Convert the model outputs for a single page image into the mmda format

        Args:
            model_outputs (lp.Layout):
                The layout detection results from the layoutparser model for
                a page image
            page_tokens (List[Span]):
                List of the Document's Token spans for this Page
            page_index (int):
                The index of the current page, used for creating the
                `Box` object
            image (PIL.Image):
                The image of the current page, used for converting
                to relative coordinates for the box objects

        Returns:
            List[SpanGroup]:
            The detected layout stored in the SpanGroup format.
        """

        final_span_groups: List[SpanGroup] = []
        id_counter = itertools.count()
        page_width, page_height = image.size

        for ele in model_outputs:
            model_output_box = Box.from_coordinates(
                x1=ele.block.x_1,
                y1=ele.block.y_1,
                x2=ele.block.x_2,
                y2=ele.block.y_2,
                page=page_index
            )

            # todo: be more precise about what's included after basic overlap
            overlapping_tokens: List[Span] = find_overlapping_tokens_for_box(page_tokens, model_output_box)
            token_boxes = [token_span.box for token_span in overlapping_tokens]

            tightened_output_box = Box.small_boxes_to_big_box(token_boxes)

            final_span_groups.append(SpanGroup(
                    spans=[
                        Span(
                            start=overlapping_tokens[0].start,
                            end=overlapping_tokens[-1].end,
                            box=tightened_output_box.get_relative(
                                page_width=page_width,
                                page_height=page_height,
                            )
                        )
                    ],
                    type=ele.type,  # TODO: it's "bibentry", but should I just omit? it's assumed that's what this predictor returns
                    id=next(id_counter)
                )
            )
        return final_span_groups

    def predict(self, document: Document) -> List[SpanGroup]:
        """Returns a list of SpanGroups for the detected layouts for all pages

        Args:
            document (Document):
                The input document object

        Returns:
            List[SpanGroup]:
                The returned SpanGroups for the detected layouts for all pages
        """
        document_prediction: List[SpanGroup] = []

        for image_index, image in enumerate(tqdm(document.images)):
            model_outputs: lp.Layout = self.model.detect(image)
            page_tokens: List[Span] = list(
                itertools.chain.from_iterable(
                    token_span_group.spans
                    for token_span_group in document.pages[image_index].tokens
                )
            )
            document_prediction.extend(
                self.postprocess(model_outputs, page_tokens, image_index, image)
            )

        return document_prediction
