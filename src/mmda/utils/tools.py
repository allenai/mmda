from __future__ import annotations
from collections import defaultdict
from itertools import groupby
from typing import List, Dict, Tuple

from mmda.types.span import Span
from mmda.types.box import Box


def allocate_overlapping_tokens_for_box(
    tokens: List["SpanGroup"], box, token_box_in_box_group: bool = False
) -> Tuple[List[Span], List[Span]]:
    """Different from `find_overlapping_tokens_for_box`, it will return a tuple
    (allocate_tokens, remaining_tokens):
    `allocated_tokens` is a list of token SpanGroups where their boxes overlap with the input box,
    `remaining_tokens` is a list of token SpanGroups where they don't overlap with the input box.
    """

    allocated_tokens, remaining_tokens = [], []
    for token in tokens:
        if token_box_in_box_group and token.box_group.boxes[0].is_overlap(box):
            # The token "box" is stored within the SpanGroup's .box_group
            allocated_tokens.append(token)
        elif token.spans[0].box is not None and token.spans[0].box.is_overlap(box):
            # default to assuming the token "box" is stored in the SpanGroup .box
            allocated_tokens.append(token)
        else:
            remaining_tokens.append(token)
    return allocated_tokens, remaining_tokens


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
        is_neighboring_spans = (
            lambda span1, span2: min(
                abs(span1.start - span2.end), abs(span1.end - span2.start)
            )
            <= self.index_distance
        )

        for i, span_i in enumerate(self.list_of_spans):
            for j in range(i + 1, len(self.list_of_spans)):
                if is_neighboring_spans(span_i, self.list_of_spans[j]):
                    self.graph[span_i.start, span_i.end].append(self.list_of_spans[j])
                    self.graph[
                        self.list_of_spans[j].start, self.list_of_spans[j].end
                    ].append(span_i)

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
