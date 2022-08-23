from copy import copy
from functools import reduce
import itertools
from typing import Union, List, Dict, Any, Optional

import layoutparser as lp

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.names import Pages, Images, Tokens
from mmda.types.span import Span


def union(block1, block2):
    x11, y11, x12, y12 = block1.coordinates
    x21, y21, x22, y22 = block2.coordinates

    return lp.Rectangle(min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))


def union_blocks(blocks):
    return reduce(union, blocks)


def make_rect(box: Box, page_width, page_height):
    box = box.get_absolute(page_width, page_height)
    rect = lp.elements.Rectangle(x_1=box.l, y_1=box.t, x_2=(box.l + box.w), y_2=(box.t + box.h))

    return rect


def tighten_boxes(bib_box_group, page_tokens, page_width, page_height):
    page_token_rects = [make_rect(span.box, page_width, page_height) for span in page_tokens]
    page_tokens_as_layout = lp.elements.Layout(blocks=page_token_rects)
    page_tokens = page_tokens_as_layout

    new_boxes = []
    for box in bib_box_group.boxes:
        abs_box = box.get_absolute(page_width, page_height)
        rect = lp.elements.Rectangle(
            abs_box.l,
            abs_box.t,
            abs_box.l + abs_box.w,
            abs_box.t + abs_box.h
        )
        new_rect = union_blocks(page_tokens.filter_by(rect, center=True))
        new_boxes.append(
            Box(l=new_rect.x_1,
                t=new_rect.y_1,
                w=new_rect.width,
                h=new_rect.height,
                page=box.page).get_relative(
                page_width=page_width,
                page_height=page_height,
            )
        )
    new_box_group = BoxGroup(
        boxes=new_boxes,
        id=bib_box_group.id,
        type="processed_box"

    )
    return new_box_group


class BibEntryDetectionPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["layoutparser", "detectron2"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Images, Tokens]

    def __init__(self, artifacts_dir: str, threshold: float = 0.88):
        label_map = {0: "bibentry"}

        self.model = lp.Detectron2LayoutModel(
            config_path=f"{artifacts_dir}/archive/config.yaml",
            model_path=f"{artifacts_dir}/archive/model_final.pth",
            label_map=label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", threshold]
        )

    def postprocess(self,
                    model_outputs: lp.Layout,
                    page_tokens: List[Span],
                    page_index: int,
                    image: "PIL.Image",
                    doc: Document) -> (List[SpanGroup], List[BoxGroup]):
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
            doc (Document):
                The document

        Returns:
            (List[SpanGroup], List[BoxGroup]):
               A tuple of the SpanGroups containing Spans of symbols within detected bibentry boxes tightened around
               tokens, the tightened box, and corresponding text, and secondly,
               the BoxGroups containing the originally detected, unprocessed bounding boxes.
        """
        id_counter = itertools.count()
        original_box_groups: List[BoxGroup] = []
        page_width, page_height = image.size

        for ele in model_outputs:
            model_output_box = Box.from_coordinates(
                x1=ele.block.x_1,
                y1=ele.block.y_1,
                x2=ele.block.x_2,
                y2=ele.block.y_2,
                page=page_index
            ).get_relative(
                page_width=page_width,
                page_height=page_height,
            )

            current_id = next(id_counter)

            # We store the original model output boxes
            original_box_groups.append(
                BoxGroup(
                    boxes=[model_output_box],
                    id=current_id,
                    type="raw_model_prediction"
                )
            )

        processed_box_groups: List[BoxGroup] = []
        for o_box_group in original_box_groups:
            tightened_box_group = tighten_boxes(o_box_group, page_tokens, page_width, page_height)
            processed_box_groups.append(tightened_box_group)

        # annotate w/ the tightened boxes to get the spans
        doc.annotate(processed_box_groups=processed_box_groups, is_overwrite=True)

        final_span_groups: List[SpanGroup] = []
        for i, sg_of_p_box_group in enumerate(doc.processed_box_groups):
            text = "".join(
                symbol for symbol in sg_of_p_box_group.symbols if sg_of_p_box_group.symbols)
            # annotating the doc with the box groups gives the corresponding SpanGroup a different ID,
            # so set them equal for easy matching
            sg_of_p_box_group.id = sg_of_p_box_group.box_group.id
            # fill in text field for the SpanGroup (for easy downstream use)
            sg_of_p_box_group.text = text
            # give them a nice type
            sg_of_p_box_group.type = "bib_entry"

            final_span_groups.append(sg_of_p_box_group)

        return final_span_groups, original_box_groups

    def predict(self, doc: Document, min_vila_bib_rows: int) -> List[SpanGroup]:
        """Returns a list of SpanGroups for the detected bibentry boxes for pages identified as bib containing pages
        via VILA heuristic (pages with "Bibliography" Vila SpanGroups that span more rows than min_vila_bib_rows).

        Args:
            doc (Document):
                The input document object containing all required annotations
            min_vila_bib_rows (int):
                Minimum number of rows in a Bibliography VILA SpanGroup required to qualify as a Bibliography section

        Returns:
            (List[SpanGroup], List[BoxGroup]):
                A tuple of the SpanGroups containing Spans of symbols within detected bibentry boxes tightened around
                tokens, the tightened box, and corresponding text, and secondly,
                the BoxGroups containing the originally detected, unprocessed bounding boxes.
        """
        document_prediction: List[SpanGroup] = []
        document_original_boxes: List[BoxGroup] = []

        vila_bib_sgs = [sg for sg in doc.vila_span_groups if
                        sg.type == "Bibliography" and (len(sg.rows) > min_vila_bib_rows)]
        vila_bib_pgs = set([sg.rows[0].spans[0].box.page for sg in vila_bib_sgs])
        vila_bib_pg_to_image = {page_index: doc.images[page_index] for page_index in vila_bib_pgs}

        for page_index, image in vila_bib_pg_to_image.items():
            model_outputs: lp.Layout = self.model.detect(image)
            page_tokens: List[Span] = list(
                itertools.chain.from_iterable(
                    token_span_group.spans
                    for token_span_group in doc.pages[page_index].tokens
                )
            )

            processed_span_groups, og_box_groups = self.postprocess(
                model_outputs,
                page_tokens,
                page_index,
                image,
                copy(doc)
            )

            document_prediction.extend(processed_span_groups)
            document_original_boxes.extend(og_box_groups)

        return document_prediction, document_original_boxes
