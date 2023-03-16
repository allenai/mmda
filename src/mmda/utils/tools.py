import json
from typing import List, Union, Dict, Any, Tuple, Optional
from collections import defaultdict

from mmda.types.span import Span
from mmda.types.box import Box


def allocate_overlapping_tokens_for_box(
        token_spans: List[Span], box
) -> Tuple[List[Span], List[Span]]:
    """Different from `find_overlapping_tokens_for_box`, it will return a tuple
    (allocate_tokens, remaining_tokens):
    `allocated_tokens` is a list of spans where their boxes overlap with the input box,
    `remaining_tokens` is a list of spans where they don't overlap with the input box.
    """

    allocated_tokens, remaining_tokens = [], []
    for token in token_spans:
        if token.box is not None and token.box.is_overlap(box):
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

    def __init__(self, list_of_spans: List["Span"], w: float = 0, h: float = 0, index_distance: int = 1) -> None:
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

    def build_graph_index_overlap(self):
        """
        Build graph, each node is represented by (start, end) of tuple, with the list of spans. Spans are considered
        overlapping if they are index_distance apart
        """
        is_neighboring_spans = (
            lambda span1, span2: min(
                abs(span1.start - span2.end), abs(span1.end - span2.start)
            ) <= self.index_distance
        )

        for i, span_i in enumerate(self.list_of_spans):
            for j in range(i + 1, len(self.list_of_spans)):
                if is_neighboring_spans(span_i, self.list_of_spans[j]):
                    self.graph[hash(json.dumps(span_i.to_json()))].append(self.list_of_spans[j])
                    self.graph[hash(json.dumps(self.list_of_spans[j].to_json()))].append(span_i)

    def build_graph_box_overlap(self):
        """
        Build graph, each node is represented by (start, end) of tuple, with the list of spans with overlapping
        boxes given, w, h
        """
        for i, span_i in enumerate(self.list_of_spans):
            assert hasattr(span_i, 'box'), 'Missing attribute box in a span'
            for j in range(i + 1, len(self.list_of_spans)):
                assert hasattr(self.list_of_spans[j], 'box'), \
                    f'Missing attribute box in a span: {self.list_of_spans[j].to_json()}'
                #if self.w == 0 and self.h == 0:
                #    self.w, self.h = (max(span_i.box.w * 0.02, self.list_of_spans[j].box.w * 0.02),
                #                      max(span_i.box.h * 0.02, self.list_of_spans[j].box.h * 0.02))
                if span_i.box.is_overlap(self.list_of_spans[j].box, self.w, self.h):
                    self.graph[hash(json.dumps(span_i.to_json()))].append(self.list_of_spans[j])
                    self.graph[hash(json.dumps(self.list_of_spans[j].to_json()))].append(span_i)

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
                span_top = stack.pop()
                node_hash = hash(json.dumps(span_top.to_json()))
                if node_hash not in visited:
                    visited.add(node_hash)
                    nodes_in_comp[comp_number].append(span_top)
                    stack.extend(self.graph[node_hash])

        # mark all nodes in the same connected component with the same integer.
        for span in self.list_of_spans:
            center = hash(json.dumps(span.to_json()))
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
                merged_box = Box.small_boxes_to_big_box([span.box for span in nodes_in_comp[comp]])
                filtered_start = [span.start for span in nodes_in_comp[comp] if span.start != -9999]
                filtered_end = [span.end for span in nodes_in_comp[comp] if span.start != -9999]
                if filtered_start:
                    min_ = min(filtered_start)
                else:
                    min_ = -9999
                if filtered_end:
                    max_ = max(filtered_end)
                else:
                    max_ = -9999

                merged_spans.append(Span(start=min_, end=max_, box=merged_box))
        return merged_spans
