from typing import List, Union, Dict, Any, Tuple, Optional
from collections import defaultdict

from mmda.types.span import Span
from mmda.types.box import Box


def merge_neighbor_spans(spans: List[Span], distance: int) -> List[Span]:
    """Merge neighboring spans in a list of un-overlapped spans:
    when the gaps between neighboring spans is not larger than the
    specified distance, they are considered as the neighbors.

    Args:
        spans (List[Span]): The input list of spans.
        distance (int, optional):
            The upper bound of interval gaps between two neighboring spans.
            Defaults to 1.

    Returns:
        List[Span]: A list of merged spans
    """

    is_neighboring_spans = (
        lambda span1, span2: min(
            abs(span1.start - span2.end), abs(span1.end - span2.start)
        )
        <= distance
    )
    # It assumes non-overlapped intervals within the list

    merge_neighboring_spans = lambda span1, span2: Span(
        min(span1.start, span2.start), max(span1.end, span2.end)
    )

    spans = sorted(spans, key=lambda ele: ele.start)
    # When sorted, only one iteration round is needed.

    if len(spans) == 0:
        return []
    if len(spans) == 1:
        return spans

    cur_merged_spans = [spans[0]]

    for cur_span in spans[1:]:
        prev_span = cur_merged_spans.pop()
        if is_neighboring_spans(cur_span, prev_span):
            cur_merged_spans.append(merge_neighboring_spans(prev_span, cur_span))
        else:
            # In this case, the prev_span should be moved to the bottom of the stack
            cur_merged_spans.extend([prev_span, cur_span])

    return cur_merged_spans


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
    Given w=width and h=height overlap neighboring spans which are w, h or less apart
    Inspired by https://leetcode.com/problems/merge-intervals/
    """
    def __init__(self, list_of_spans: List["Span"], w: float, h: float) -> None:
        self.list_of_spans = list_of_spans
        self.w = w
        self.h = h
        self.graph = defaultdict(list)

    def build_graph(self):
        """
        Build graph, each node is represented by (start, end) of tuple, with the list of spans with overlapping
        boxes given, w, h
        """
        for i, span_i in enumerate(self.list_of_spans):
            for j in range(i + 1, len(self.list_of_spans)):
                if span_i.box.is_overlap(self.list_of_spans[j].box, self.w, self.h):
                    self.graph[span_i.start, span_i.end].append(self.list_of_spans[j])
                    self.graph[self.list_of_spans[j].start, self.list_of_spans[j].end].append(span_i)

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

    def merge(self):
        """
        For each of the lists of the connected nodes, merge boxes and find, min, max of the index
        """
        self.build_graph()
        nodes_in_comp, number_of_comps = self.get_components()

        # all intervals in each connected component must be merged.
        merged_spans = []
        for comp in range(number_of_comps):
            if nodes_in_comp[comp]:
                merged_box = Box.small_boxes_to_big_box([span.box for span in nodes_in_comp[comp]])
                merged_spans.append(Span(start=min([span.start for span in nodes_in_comp[comp]]),
                                         end=max([span.end for span in nodes_in_comp[comp]]), box=merged_box))
        return merged_spans
