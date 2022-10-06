from collections import defaultdict

from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

from ai2_internal import api
from mmda.predictors.base_predictors.base_heuristic_predictor import BaseHeuristicPredictor
from mmda.types import SpanGroup, BoxGroup
from mmda.types.document import Document
from mmda.types.span import Span
from mmda.utils.tools import MergeSpans


class FigureTablePredictions(BaseHeuristicPredictor):
    REQUIRED_DOCUMENT_FIELDS = ['pages', 'tokens', 'rows', 'layoutparser_span_groups', 'vila_span_groups']

    @staticmethod
    def merge_spans(vila_span_groups: List[api.SpanGroup], caption_content: str = 'fig') -> Dict[int, List[api.Box]]:
        """
        Merges span_groups if overlapping
        """
        vila_caption = [span_group for span_group in vila_span_groups if
                        span_group.type == 'Caption' and caption_content in span_group.text.replace(' ', '').lower()]

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
    def _merge_boxes(layoutparser_span_groups: List[api.SpanGroup], types: List[str] = ['Figure']):
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
    def _get_object_caption_distance(figure_box, caption_box) -> float:
        """
        Return 900.0 if left point of figure, caption is offset more than 10%
        Otherwise returns distance between top of the figure box and cap box
        """
        l_fig, t_fig = figure_box.l + figure_box.w / 2, figure_box.t + figure_box.h / 2
        l_cap, t_cap = caption_box.l + caption_box.w / 2, caption_box.t + caption_box.h
        if abs(l_fig - l_cap) / l_fig > 0.1:
            return 900.0

        return t_cap - t_fig

    @staticmethod
    def _predict(doc: Document, caption_type: str = 'Figure') -> List[SpanGroup]:
        """
        Merges boxes corresponding to tokens of table, figure captions. For each page each caption/object create cost
        matrix which is distance based on get_object_caption_distance. Using linear_sum_assignment find corresponding
        pairs, caption-object
        """
        assert doc.layoutparser_span_groups
        assert doc.vila_span_groups
        if caption_type == 'Figure':
            merged_boxes_fig_dict = FigureTablePredictions._merge_boxes(doc.layoutparser_span_groups)
            merged_boxes_caption_dict = FigureTablePredictions.merge_spans(doc.vila_span_groups)
        else:
            merged_boxes_fig_dict = FigureTablePredictions._merge_boxes(doc.layoutparser_span_groups, types=['Table'])
            merged_boxes_caption_dict = FigureTablePredictions.merge_spans(doc.vila_span_groups, caption_content='tab')

        predictions = []
        for page in range(len(tqdm(doc.images))):
            if merged_boxes_caption_dict[page] and merged_boxes_fig_dict[page]:
                cost_matrix = np.zeros((len(merged_boxes_fig_dict[page]),
                                        len(merged_boxes_caption_dict[page])))
                for j, fig_box in enumerate(merged_boxes_fig_dict[page]):
                    for i, caption_box in enumerate(merged_boxes_caption_dict[page]):
                        assert hasattr(fig_box, 'box')
                        assert hasattr(caption_box, 'box')
                        distance = FigureTablePredictions._get_object_caption_distance(
                            fig_box.box, caption_box.box)

                        if caption_type == 'Figure':
                            cost_matrix[j][i] = distance if distance > 0 else 900
                        else:
                            cost_matrix[j][i] = distance if distance < 0 else 900

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for row, col in zip(row_ind, col_ind):
                    predictions.append(SpanGroup(spans=[Span(
                        start=merged_boxes_caption_dict[page][col].start,
                        end=merged_boxes_caption_dict[page][col].end,
                        box=merged_boxes_caption_dict[page][col].box)],
                        box_group=BoxGroup(boxes=[merged_boxes_fig_dict[page][row].box], id=None, type=caption_type),
                        id=None,
                        type=caption_type,
                        text=doc.symbols[merged_boxes_caption_dict[page][col].start:
                                         merged_boxes_caption_dict[page][col].end]
                    ))
        return predictions

    @staticmethod
    def predict(document: Document) -> Tuple[List[SpanGroup], List[SpanGroup]]:
        """
        Predict figure->caption mapping and, table->caption mapping.
        Return: List[SpanGroup], SpanGroup has start, end corresponding to caption start, end indexes and box
        corresponding to merged boxes of the tokens of the caption. Type is one of ['Figure', 'Table']. BoxGroup stores
        information about the boundaries of figure or table.
        """
        predictions = []
        predictions.extend(self._predict(document, caption_type='Figure'))
        predictions.extend(self._predict(document, caption_type='Table'))

        return predictions
