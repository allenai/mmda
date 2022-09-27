import json
from collections import defaultdict

from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from dataclasses import dataclass

from tqdm import tqdm

from ai2_internal import api
from mmda.predictors.base_predictors.base_heuristic_predictor import BaseHeuristicPredictor
from mmda.types.document import Document
from mmda.predictors.hf_predictors.token_classification_predictor import IVILATokenClassificationPredictor
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.utils.tools import MergeSpans


@dataclass
class ObjectCaptionMap:
    dpi: int
    page: int
    object_type: str
    box: api.Box
    caption: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class FigureTablePredictions(BaseHeuristicPredictor):
    REQUIRED_BACKENDS = ['layoutparser', 'vila']

    def __init__(self, dpi: int = 72):
        self.doc = None
        self.dpi = dpi

    def create_doc_rasterize(self, pdf_path):
        self.doc = PDFPlumberParser().parse(input_pdf_path=pdf_path)
        assert self.doc.pages
        assert self.doc.tokens
        images = PDF2ImageRasterizer().rasterize(input_pdf_path=pdf_path, dpi=self.dpi)
        self.doc.annotate_images(images=images)

    def make_vision_prediction(self, doc):
        """

        """
        vision_predictor = LayoutParserPredictor.from_pretrained()
        layoutparser_span_groups = vision_predictor.predict(document=doc)
        doc.annotate(layoutparser_span_groups=layoutparser_span_groups)
        return doc

    def make_villa_predictions(
            self, doc, model_name: str = 'allenai/ivila-row-layoutlm-finetuned-s2vl-v2') -> Dict[int, List[api.Box]]:
        """
        """
        vila_predictor = IVILATokenClassificationPredictor.from_pretrained(model_name)
        vila_span_groups = vila_predictor.predict(document=doc)
        doc.annotate(vila_span_groups=vila_span_groups)
        return doc

    @staticmethod
    def merge_spans(vila_span_groups: List[api.SpanGroup]) -> Dict[int, List[api.Box]]:
        """
        """
        vila_caption = [span_group for span_group in vila_span_groups if
                        span_group.type == 'Caption' and 'fig' in span_group.text.replace(' ', '').lower()]

        vila_caption_dict = defaultdict(list)
        width_heights_dict = defaultdict(list)
        for entry_caption in vila_caption:
            for span_group in entry_caption.tokens:
                for span in span_group.spans:
                    vila_caption_dict[span.box.page].append(span)
                    width_heights_dict[span.box.page].append([span.box.w, span.box.h])

        merged_boxes_list = defaultdict(list)

        for page, list_of_boxes in vila_caption_dict.items():
            # Calculate average token w, h on the page
            w_avg, h_avg = np.average(width_heights_dict[page], axis=0)
            merged_boxes_list[page] = MergeSpans(list_of_spans=list_of_boxes, w=w_avg * 1.2,
                                                 h=h_avg).merge_neighbor_spans_by_box_coordinate()

        return merged_boxes_list

    @staticmethod
    def merge_boxes(layoutparser_span_groups: List[api.SpanGroup], types: List[str] = ['Figure']):
        """
        Merges overlapping boxes
        """
        span_map = defaultdict(list)
        for span in layoutparser_span_groups:
            if span.box_group.type in types:
                for box in span.box_group.boxes:
                    # Creating unique start, end of spans used as a key for merging boxes
                    box_api = api.Box.from_mmda(box)
                    span_map[box.page].append(api.Span(start=int(box_api.left * 1000),
                                                       end=int(box_api.left * 1000 + 10),
                                                       box=box_api).to_mmda())

        return {page: MergeSpans(span_map[page], w=0, h=0).merge_neighbor_spans_by_box_coordinate() for page in
                span_map.keys()}

    @staticmethod
    def get_figure_caption_distance(figure_box, caption_box):
        """
        """
        l_fig, t_fig = figure_box.l + figure_box.w / 2, figure_box.t + figure_box.h / 2
        l_cap, t_cap = caption_box.l + caption_box.w / 2, caption_box.t + caption_box.h
        if abs(l_fig - l_cap) / l_fig > 0.1:
            return 900

        return t_cap - t_fig if t_cap - t_fig > 0 else 900

    def make_boxgroups(self, doc: Document, page: int, box: api.Box) -> Dict[int, List[Tuple[float, float, float,
                                                                                             float]]]:
        """
        """
        page_w, page_h = doc.images[page].size
        width_height = [page_w, page_h, page_w, page_h]
        coordinates = box.coordinates
        return [coordinates[idx] * width_height[idx] for idx in range(4)]

    def predict(self, doc: Document) -> Dict[int, List[ObjectCaptionMap]]:
        """
        """
        assert doc.layoutparser_span_groups

        merged_boxes_fig_dict = self.merge_boxes(doc.layoutparser_span_groups)
        self.make_villa_predictions(doc)
        merged_boxes_caption_dict = self.merge_spans(doc.vila_span_groups)

        predictions = []
        for page in range(len(tqdm(doc.images))):
            if merged_boxes_caption_dict[page] and merged_boxes_fig_dict[page]:
                cost_matrix = np.zeros((len(merged_boxes_fig_dict[page]), len(merged_boxes_caption_dict[page])))
                for j, fig_box in enumerate(merged_boxes_fig_dict[page]):
                    for i, caption_box in enumerate(merged_boxes_caption_dict[page]):
                        assert hasattr(fig_box, 'box')
                        assert hasattr(caption_box, 'box')
                        cost_matrix[j][i] = self.get_figure_caption_distance(fig_box.box,
                                                                             caption_box.box)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for row, col in zip(row_ind, col_ind):
                    predictions.append(ObjectCaptionMap(
                        self.dpi, page, 'Figure', self.make_boxgroups(doc, page, merged_boxes_fig_dict[page][row].box),
                        doc.symbols[merged_boxes_caption_dict[page][col].start:
                                    merged_boxes_caption_dict[page][col].end]))

        return predictions
