from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ai2_internal import api
from ai2_internal.api import Relation
from mmda.predictors.base_predictors.base_heuristic_predictor import (
    BaseHeuristicPredictor,
)
from mmda.types import BoxGroup, SpanGroup
from mmda.types.document import Document
from mmda.types.span import Span


class MergeSpans:
    """
    Given w=width and h=height merge neighboring spans which are w, h or less apart or by merging neighboring spans
    which are index distance apart
    Inspired by https://leetcode.com/problems/merge-intervals/
    """

    def __init__(
        self,
        list_of_spans: List["Span"],
        w: float = 0,
        h: float = 0,
        index_distance: int = 1,
    ) -> None:
        """
        Args
            w (float): The input width between boxes to merge
            h (float): The input height between the boxes to merge
            index_distance (int): Distance between the spans
        """
        self.list_of_spans = list_of_spans
        self.w = w
        self.h = h
        self.graph = defaultdict(list)
        self.index_distance = index_distance

    @classmethod
    def from_span_groups_with_box_groups(
        cls,
        span_groups: List["SpanGroup"],
        w: float = 0,
        h: float = 0,
        index_distance: int = 1,
    ) -> MergeSpans:
        # Convert SpanGroups with single box_group box into SpanGroups with span.box
        spans_with_boxes = []
        for sg in span_groups:
            assert len(sg.spans) == len(
                sg.box_group.boxes
            ), "Unequal number of spans and boxes for SpanGroup"
            for span, box in zip(sg.spans, sg.box_group.boxes):
                spans_with_boxes.append(Span(start=span.start, end=span.end, box=box))
        return cls(spans_with_boxes, w, h, index_distance)

    def build_graph_index_overlap(self):
        """
        Build graph, each node is represented by (start, end) of tuple, with the list of spans. Spans are considered
        overlapping if they are index_distance apart
        """
        starts_matrix = np.full(
            (len(self.list_of_spans), len(self.list_of_spans)),
            [span.start for span in self.list_of_spans],
        )
        ends_matrix = np.full(
            (len(self.list_of_spans), len(self.list_of_spans)),
            [span.end for span in self.list_of_spans],
        )

        starts_minus_ends = np.abs(starts_matrix - ends_matrix.T)
        ends_minus_starts = np.abs(ends_matrix - starts_matrix.T)
        are_neighboring_spans = (
            np.minimum(starts_minus_ends, ends_minus_starts) <= self.index_distance
        )
        neighboring_spans = np.transpose(are_neighboring_spans.nonzero())

        if len(neighboring_spans) > 0:
            neighboring_spans_no_dupes = neighboring_spans[
                np.where(neighboring_spans[:, 1] < neighboring_spans[:, 0])
            ]

            for j, i in neighboring_spans_no_dupes:
                span_i = self.list_of_spans[i]
                span_j = self.list_of_spans[j]
                self.graph[span_i.start, span_i.end].append(span_j)
                self.graph[span_j.start, span_j.end].append(span_i)

    def build_graph_box_overlap(self):
        """
        Build graph, each node is represented by (start, end) of tuple, with the list of spans with overlapping
        boxes given, w, h
        """
        for i, span_i in enumerate(self.list_of_spans):
            assert hasattr(span_i, "box"), "Missing attribute box in a span"
            for j in range(i + 1, len(self.list_of_spans)):
                assert hasattr(
                    self.list_of_spans[j], "box"
                ), "Missing attribute box in a span"
                if span_i.box.is_overlap(self.list_of_spans[j].box, self.w, self.h):
                    self.graph[span_i.start, span_i.end].append(self.list_of_spans[j])
                    self.graph[
                        self.list_of_spans[j].start, self.list_of_spans[j].end
                    ].append(span_i)

    # gets the connected components of the boxes overlap graph.
    def get_components(self):
        """
        Groups connected graph nodes into dictionary list
        """
        visited = set()
        comp_number = 0
        nodes_in_comp = defaultdict(list)

        def mark_component_dfs(start):
            stack = [start]
            while stack:
                span = stack.pop()
                node = span.start, span.end
                if node not in visited:
                    visited.add(node)
                    nodes_in_comp[comp_number].append(span)
                    stack.extend(self.graph[node])

        # mark all nodes in the same connected component with the same integer.
        for span in self.list_of_spans:
            center = span.start, span.end
            if center not in visited:
                mark_component_dfs(span)
                comp_number += 1

        return nodes_in_comp, comp_number

    def merge_neighbor_spans_by_symbol_distance(self):
        """
        For each of the lists of the connected nodes determined by index distance between the spans,
        merge boxes and find, min, max of the index
        """
        return self.build_merged_spans_from_connected_components(index=True)

    def merge_neighbor_spans_by_box_coordinate(self):
        """
        For each of the lists of the connected nodes determined by distance between the boxes,
        merge boxes and find, min, max of the index
        """
        return self.build_merged_spans_from_connected_components(index=False)

    def build_merged_spans_from_connected_components(self, index):
        """
        For each of the lists of the connected nodes determined by symbol distance or box distance,
        merge boxes and find, min, max of the index
        """
        if index:
            self.build_graph_index_overlap()
        else:
            self.build_graph_box_overlap()

        nodes_in_comp, number_of_comps = self.get_components()

        # all intervals in each connected component must be merged.
        merged_spans = []
        for comp in range(number_of_comps):
            if nodes_in_comp[comp]:
                spans_by_page: Dict[any, List[Span]] = defaultdict(list)
                for pg, page_spans in groupby(
                    nodes_in_comp[comp],
                    lambda s: s.box.page if s.box is not None else None,
                ):
                    for span in page_spans:
                        spans_by_page[pg].append(span)
                for page_spans in spans_by_page.values():
                    merged_box = Box.small_boxes_to_big_box(
                        [span.box for span in page_spans]
                    )
                    merged_spans.append(
                        Span(
                            start=min([span.start for span in page_spans]),
                            end=max([span.end for span in page_spans]),
                            box=merged_box,
                        )
                    )
        return merged_spans


class FigureTablePredictions(BaseHeuristicPredictor):
    """Class for creating a map of figure boxes to figure captions"""

    REQUIRED_DOCUMENT_FIELDS = [
        "pages",
        "tokens",
        "vila_span_groups",
        "blocks",
    ]

    def __init__(self, document: Document) -> None:
        self.doc = document
        self.vila_caption_dict = None
        self.vila_spans_all_dict = None
        self.width_heights_dict = None
        self.w_avg, self.h_avg = FigureTablePredictions.get_avg_w_h_of_tokens(
            self.doc.tokens
        )
        # Parameteer for the fraction of the tokens classified as non-caption that are probably caption in same
        # Layoutparser span group
        self.FRACTION_OF_MISCLASSIFIED_VILA_CAPTION_TOKENS = 0.3

    @staticmethod
    def _create_dict_of_pages_spans_vila(span_groups=None) -> Dict[int, List[Span]]:
        """
        Create a dictionary of page number to list of spans
        Returns: Dict[int, List[Span]] dictionary of page number to list of spans
        """
        vila_dict = defaultdict(list)
        for entry_caption in span_groups:
            for token in entry_caption.tokens:
                for span in token.spans:
                    span.span_id = token.id
                    span.type = entry_caption.type
                    vila_dict[span.box.page].append(span)
        return vila_dict

    @staticmethod
    def get_avg_w_h_of_tokens(tokens) -> Tuple[float, float]:
        """
        Get the average width and height of tokens
        """
        return np.average(
            [[span.box.w, span.box.h] for token in tokens for span in token.spans],
            axis=0,
        )

    @staticmethod
    def _create_dict_of_pages_spans_layoutparser(
        layoutparser_span_groups,
        types: List[str] = [],
        starts_with: str = "",
        negation: bool = False,
    ) -> Dict[int, List[SpanGroup]]:
        """
        Create a dictionary of page number to list of spans, filtering or negating to the types and starts_with
        """
        span_map = defaultdict(list)
        for span_group in layoutparser_span_groups:
            if not types or span_group.box_group.type in types:
                if negation:
                    starts_with_bool = not span_group.text.lower().startswith(
                        starts_with
                    )
                else:
                    starts_with_bool = span_group.text.lower().startswith(starts_with)

                if starts_with_bool:
                    for box in span_group.box_group.boxes:
                        # Creating unique start, end of spans used as a key for merging boxes
                        box_api = api.Box.from_mmda(box)
                        if span_group.spans and len(span_group.spans) == 1:
                            start, end = (
                                span_group.spans[0].start,
                                span_group.spans[0].end,
                            )
                        else:
                            start, end = -9999, -9999

                        created_span = api.Span(
                            start=start, end=end, box=box_api
                        ).to_mmda()

                        created_span.span_id = span_group.id
                        created_span.box_group_type = span_group.box_group.type
                        span_map[box.page].append(created_span)
                    # Bring in the boxes from the span groups
                    for span in span_group.spans:
                        box_api = api.Box.from_mmda(span.box)
                        created_span = api.Span(
                            start=span.start, end=span.end, box=box_api
                        ).to_mmda()
                        # Note that hash output is changing everytime it is called
                        created_span.span_id = f"LP_span_group_{span.box.page}_{len(span_map[span.box.page])}"
                        created_span.box_group_type = span_group.box_group.type
                        span_map[span.box.page].append(created_span)
        return span_map

    @staticmethod
    def generate_map_of_layout_to_tokens(
        vila_dict, layout_parser_overlap, dict_of_pages_layoutparser, key="caption"
    ) -> Dict[int, Dict]:
        """
        Generate a map of layoutparser entries to the list of vila tokens with the type = key vs type != key
        """
        for page in vila_dict.keys():
            for span in vila_dict[page]:
                for layout_span in dict_of_pages_layoutparser.get(page, []):
                    if span.box.is_overlap(layout_span.box):
                        id_dict = layout_parser_overlap.get(
                            layout_span.span_id, {"caption": [], "non_caption": []}
                        )
                        id_dict[key].append(span.span_id)
                        layout_parser_overlap[layout_span.span_id] = id_dict
        return layout_parser_overlap

    @staticmethod
    def generate_map_of_layout_to_tokens_for_page(
        vila_list: List, layout_parser_list: List, key="caption"
    ) -> Dict[int, Dict]:
        """
        Generate a map of layoutparser tokens ids to the count of vila tokens with the type = key
        """
        layout_parser_overlap = dict()
        for span in vila_list:
            for layout_span in layout_parser_list:
                if span.box.is_overlap(layout_span.box):
                    id_dict = layout_parser_overlap.get(
                        layout_span.span_id, {"caption": [], "non_caption": []}
                    )
                    if span.type.lower() == key:
                        id_dict[key].append(span.span_id)
                    else:
                        id_dict["non_caption"].append(span.span_id)
                    layout_parser_overlap[layout_span.span_id] = id_dict
        return layout_parser_overlap

    def update_vila_caption_dict(
        self,
        vila_caption_dict: Dict[int, List[Span]],
        vila_non_caption_dict: Dict[int, List[Span]],
    ) -> Dict[int, List[Span]]:
        """
        Update the vila caption dict to cast tokens that are misclassified as no captions in ths same LayoutParser
        region
        """
        layout_parser_overlap = defaultdict(dict)
        # Build overlap map between layoutparser and caption tokens
        span_map = FigureTablePredictions._create_dict_of_pages_spans_layoutparser(
            self.doc.blocks
        )
        layout_parser_overlap = FigureTablePredictions.generate_map_of_layout_to_tokens(
            vila_caption_dict, layout_parser_overlap, span_map
        )
        # Build overlap map between layoutparser and non-caption tokens
        layout_parser_overlap = FigureTablePredictions.generate_map_of_layout_to_tokens(
            vila_non_caption_dict, layout_parser_overlap, span_map, key="non_caption"
        )

        for key, value in layout_parser_overlap.items():
            caption_token_fraction = len(value["caption"]) / (
                len(value["caption"]) + len(value["non_caption"])
            )
            if (1.0 > caption_token_fraction) and (
                caption_token_fraction
                > self.FRACTION_OF_MISCLASSIFIED_VILA_CAPTION_TOKENS
            ):
                for span_id in layout_parser_overlap[key]["non_caption"]:
                    for page, vila_span in vila_non_caption_dict.items():
                        for entry in vila_span:
                            if entry.span_id == span_id:
                                vila_caption_dict[entry.box.page].append(entry)

        return vila_caption_dict

    @staticmethod
    def _filter_span_group(
        vila_span_groups: List[api.SpanGroup],
        caption_content: str,
        span_group_types: List[str],
        negation=False,
    ) -> List[api.SpanGroup]:
        """
        Helper function which filters out span groups based on the caption content and span group type
        """
        result = []
        for span_group in vila_span_groups:
            if span_group.text.replace(" ", "").lower().startswith(caption_content):
                if span_group.type in span_group_types and not negation:
                    result.append(span_group)
                elif negation and span_group.type not in span_group_types:
                    result.append(span_group)
        return result

    def merge_vila_token_spans(
        self, caption_content: str = "fig", span_group_type: List[str] = ["Caption"]
    ) -> Dict[int, List[api.Box]]:
        """
        Merging spanGroups
        Args:
            vila_span_groups (List[api.SpanGroup]): list of span groups from vila to merge
            caption_content (str): Caption should contain caption_content value

        Returns: Dictionary page -> List of merged boxes
        """
        vila_span_groups_filtered = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups,
            caption_content=caption_content,
            span_group_types=span_group_type,
        )

        vila_caption_dict = defaultdict(list)
        for entry_caption in vila_span_groups_filtered:
            for span_group in entry_caption.tokens:
                for span in span_group.spans:
                    vila_caption_dict[span.box.page].append(span)

        merged_boxes_list = defaultdict(list)
        for page, list_of_boxes in vila_caption_dict.items():
            # Merge spans if they are sufficiently close to each other
            merged_boxes_list[page] = MergeSpans(
                list_of_spans=list_of_boxes, w=self.w_avg * 1.5, h=self.h_avg * 1
            ).merge_neighbor_spans_by_box_coordinate()
        return merged_boxes_list

    def _cast_to_caption_vila_tokens(self, caption_content="fig"):
        """
        Heuristic logic for fixing miss classified tokens as non caption. By checking layoutparser box predictions and
        tokens which belong to them, I cast the rest of the tokens to caption category.
        Args:
            layoutparser_span_groups (List[api.SpanGroup]): list of span groups from layoutparser
            vila_span_groups (List[api.SpanGroup]): list of span groups from vila to merge
        Returns (List[List[api.SpanGroup]]) list of lists of spangroups which are cast to
        """
        # First let's go over all the tokens which are labeled as caption and find the LayoutParser SpanGroups which
        # they overlap with
        vila_caption = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups,
            caption_content=caption_content,
            span_group_types=["Caption"],
        )

        self.vila_caption_dict = (
            FigureTablePredictions._create_dict_of_pages_spans_vila(vila_caption)
        )

        vila_non_caption = FigureTablePredictions._filter_span_group(
            self.doc.vila_span_groups,
            caption_content="",
            span_group_types=["Caption"],
            negation=True,
        )

        vila_non_caption_dict = FigureTablePredictions._create_dict_of_pages_spans_vila(
            vila_non_caption
        )
        return self.update_vila_caption_dict(
            self.vila_caption_dict, vila_non_caption_dict
        )

    def merge_boxes(
        self,
        layoutparser_span_groups: List[api.SpanGroup],
        merged_boxes_vila_dict: Dict[int, List[api.Box]] = None,
        types: List[str] = ["Figure"],
    ) -> Dict[int, List[api.Box]]:
        """
        Merges overlapping boxes. Vila caption predictions is more consistent than layout parser prediction, thus
        we check the number of items after the merge with the number of caption boxes.
        Args:
            layoutparser_span_groups (List[api.SpanGroup]): list of span groups from layoutparser
            merged_boxes_vila_dict (List[api.SpanGroup]): list of span groups for the merged vila tokens assigned to
            the class caption
            types (List[str]): List of types of the spangroups to merge

        Returns: Dictionary of the merged figure boxes.

        """
        if merged_boxes_vila_dict is None:
            merged_boxes_vila_dict = defaultdict(list)

        merged_boxes_vila_dict_left = defaultdict(list)
        merged_boxes_map = defaultdict(list)
        span_map = FigureTablePredictions._create_dict_of_pages_spans_layoutparser(
            layoutparser_span_groups, types=types
        )

        for page, span_list in span_map.items():
            # Adding vila spans to the layout parser list of the spans
            if merged_boxes_vila_dict[page]:
                span_list.extend(merged_boxes_vila_dict_left[page])
            merged_spans = MergeSpans(
                span_list, w=self.w_avg * 0.5, h=self.h_avg * 1.0
            ).merge_neighbor_spans_by_box_coordinate()

            # Filtering out vila spans (not merged)
            if (
                len(span_list) != len(merged_spans)
                and merged_boxes_vila_dict
                and merged_boxes_vila_dict[page]
            ):
                merged_spans = [
                    merged_span
                    for merged_span in merged_spans
                    if not any(
                        vila_span.box.to_json() == merged_span.box.to_json()
                        for vila_span in merged_boxes_vila_dict[page]
                    )
                ]

                merged_boxes_vila_dict_left[page] = [
                    vila_span
                    for vila_span in merged_boxes_vila_dict[page]
                    if any(
                        vila_span.box.to_json() == merged_span.box.to_json()
                        for merged_span in merged_spans
                    )
                ]

            if merged_boxes_vila_dict_left[page]:
                merged_boxes_vila_dict[page] = merged_boxes_vila_dict_left[page]
            merged_boxes_map[page] = merged_spans

        return merged_boxes_map, merged_boxes_vila_dict

    @staticmethod
    def _get_object_caption_distance(
        figure_box: api.Box, caption_box: api.Box
    ) -> float:
        """
        Return 900.0 if left point of figure, caption is offset more than 10%
        Otherwise returns distance middle of the figure box and caption box
        Args:
            figure_box (api.Box): Box corresponding to figure
            caption_box (api.Box): Box corresponding to caption

        Returns: Distance between center of the box and caption location
        """
        l_fig, t_fig = figure_box.l + figure_box.w / 2, figure_box.t + figure_box.h / 2
        l_cap, t_cap = caption_box.l + caption_box.w / 2, caption_box.t + caption_box.h
        if abs(l_fig - l_cap) / l_fig > 0.1:
            return 900.0

        return t_cap - t_fig

    def get_layout_span_groups_starts_with(
        self, caption_content: str = "fig", vila_spans: dict = None
    ):
        """ """
        spans_to_merge_dict = defaultdict(list)
        self.vila_caption_dict = self._cast_to_caption_vila_tokens(
            caption_content=caption_content
        )

        if vila_spans:
            for page_idx, vila_span in vila_spans.items():
                spans_to_merge_dict[page_idx].extend(vila_span)

        layout_parser_span_groups_dict = defaultdict(list)

        if vila_spans:
            for page_idx, vila_span in vila_spans.items():
                layout_parser_span_groups_dict[page_idx].extend(vila_span)
        return layout_parser_span_groups_dict

    def generate_candidates(self) -> Tuple[Union[SpanGroup, BoxGroup]]:
        """
        Generates candidates for the figure and table captions
        """
        assert self.doc.vila_span_groups

        merged_boxes_caption_fig_tab_dict = {}
        for caption_content in ["fig", "tab"]:
            # Merge vila tokens which start with caption_content
            merged_boxes_caption_fig_tab_dict[
                caption_content
            ] = self.merge_vila_token_spans(caption_content=caption_content)

            merged_boxes_caption_fig_tab_dict[
                caption_content
            ] = self.get_layout_span_groups_starts_with(
                caption_content=caption_content,
                vila_spans=merged_boxes_caption_fig_tab_dict[caption_content],
            )

            # Final check that the defined captions are starting with tab and fig
            for page_idx, list_of_spans in merged_boxes_caption_fig_tab_dict[
                caption_content
            ].items():
                for span in list_of_spans:
                    if (
                        not self.doc.symbols[span.start : span.end]
                        .lower()
                        .startswith(caption_content)
                    ):
                        list_of_spans.remove(span)
                merged_boxes_caption_fig_tab_dict[caption_content][
                    page_idx
                ] = list_of_spans

        # merged_boxes_vila_dict is used in figure, table boxes derivation
        merged_boxes_vila_dict = self.merge_vila_token_spans(
            caption_content="", span_group_type=["Text", "Paragraph", "Table", "Figure"]
        )
        # Create dictionary of layoutparser span groups merging boxgroups and boxes
        merged_boxes_vila_dict_left = None
        merged_boxes_fig_tab_dict = {}
        # List of types to be merged from layoutparser, note that sometimes figures are marked as Equations
        for layout_parser_box_type in [["Figure"], ["Table"]]:
            merged_boxes_vila_dict = (
                merged_boxes_vila_dict_left
                if merged_boxes_vila_dict_left is not None
                else merged_boxes_vila_dict
            )
            (
                merged_boxes_fig_tab_dict[layout_parser_box_type[0]],
                merged_boxes_vila_dict_left,
            ) = self.merge_boxes(
                layoutparser_span_groups=self.doc.blocks,
                types=layout_parser_box_type,
                merged_boxes_vila_dict=merged_boxes_vila_dict,
            )

        return (
            merged_boxes_caption_fig_tab_dict["fig"],
            merged_boxes_fig_tab_dict["Figure"],
            merged_boxes_caption_fig_tab_dict["tab"],
            merged_boxes_fig_tab_dict["Table"],
        )

    def _predict(
        self, merged_boxes_caption_dict, merged_boxes_fig_tab_dict, caption_type
    ) -> Dict[str, Union[SpanGroup, BoxGroup, Relation]]:
        """
        Merges boxes corresponding to tokens of table, figure captions. For each page each caption/object create cost
        matrix which is distance based on get_object_caption_distance. Using linear_sum_assignment find corresponding
        pairs, caption-object
        Args:
            doc (Document): document to make predictions on, it has to have fields layoutparser_span_groups and
            vila_span_groups
            caption_type (str): caption type to make prediction for can be Figure or Table
        Returns: Returns dictionary with keys 'predictions', 'predictions_captions', 'predictions_relations'

        """
        predictions = []
        predictions_captions = []
        predictions_relations = []
        for page in range(len(tqdm(self.doc.pages))):
            if merged_boxes_caption_dict.get(page) and merged_boxes_fig_tab_dict.get(
                page
            ):
                cost_matrix = np.zeros(
                    (
                        len(merged_boxes_fig_tab_dict[page]),
                        len(merged_boxes_caption_dict[page]),
                    )
                )
                for j, fig_box in enumerate(merged_boxes_fig_tab_dict[page]):
                    for i, span_group in enumerate(merged_boxes_caption_dict[page]):
                        caption_box = span_group.box
                        assert hasattr(fig_box, "box")
                        distance = FigureTablePredictions._get_object_caption_distance(
                            fig_box.box, caption_box
                        )

                        cost_matrix[j][i] = distance
                        if caption_type == "Figure":
                            cost_matrix[j][i] = distance if distance > 0 else 900

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for row, col in zip(row_ind, col_ind):
                    # Check that caption starts with tab or fig
                    if (
                        self.doc.symbols[
                            merged_boxes_caption_dict[page][col]
                            .start : merged_boxes_caption_dict[page][col]
                            .end
                        ]
                        .lower()
                        .startswith(caption_type.lower()[:3])
                    ):
                        span_group = SpanGroup(
                            spans=[
                                Span(
                                    start=merged_boxes_caption_dict[page][col].start,
                                    end=merged_boxes_caption_dict[page][col].end,
                                )
                            ],
                            id=len(predictions_captions),
                        )
                        box_group = BoxGroup(
                            boxes=[merged_boxes_fig_tab_dict[page][row].box],
                            id=len(predictions),
                        )
                        predictions.append(box_group)
                        predictions_captions.append(span_group)
                        predictions_relations.append(
                            Relation(from_id=box_group.id, to_id=span_group.id)
                        )
        return {
            f"{caption_type.lower()}s": predictions,
            f"{caption_type.lower()}_captions": predictions_captions,
            f"{caption_type.lower()}_to_{caption_type.lower()}_captions": predictions_relations,
        }

    def predict(self) -> Dict[str, Union[SpanGroup, BoxGroup, Relation]]:
        """
        Returns: Dictionary List[SpanGroup], SpanGroup has start, end corresponding to caption start, end indexes and box
        corresponding to merged boxes of the tokens of the caption. Type is one of ['Figure', 'Table']. BoxGroup stores
        information about the boundaries of figure or table. Relation stores information about the relation between
        caption and the object it corresponds to
        """
        (
            merged_boxes_caption_fig_dict,
            merged_boxes_fig_dict,
            merged_boxes_caption_tab_dict,
            merged_boxes_tab_dict,
        ) = self.generate_candidates()
        result_dict = {}
        result_dict.update(
            self._predict(
                merged_boxes_caption_fig_dict,
                merged_boxes_fig_dict,
                caption_type="Figure",
            )
        )
        result_dict.update(
            self._predict(
                merged_boxes_caption_tab_dict,
                merged_boxes_tab_dict,
                caption_type="Table",
            )
        )
        return result_dict
