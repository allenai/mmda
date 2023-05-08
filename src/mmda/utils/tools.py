from __future__ import annotations

import logging
from collections import defaultdict
from itertools import groupby
import itertools
from typing import List, Dict, Tuple

import numpy as np

from mmda.types.annotation import BoxGroup, SpanGroup
from mmda.types.box import Box
from mmda.types.span import Span


def allocate_overlapping_tokens_for_box(
        tokens: List[SpanGroup], box, token_box_in_box_group: bool = False, x: float = 0.0, y: float = 0.0, center: bool = False
) -> Tuple[List[Span], List[Span]]:
    """Finds overlap of tokens for given box
    Args
        `tokens` (List[SpanGroup])
        `box` (Box)
        `token_box_in_box_group` (bool) defaults to False, assumes token SpanGroup box is on the single span's box.
        `center` (bool) if set to True, considers a token to be overlapping with given box only if its center overlaps
    Returns
        a tuple: (allocated_tokens, remaining_tokens):
        `allocated_tokens` is a list of token SpanGroups where their boxes overlap with the input box,
        `remaining_tokens` is a list of token SpanGroups where they don't overlap with the input box.
    """
    allocated_tokens, remaining_tokens = [], []
    for token in tokens:
        if token_box_in_box_group and token.box_group.boxes[0].is_overlap(other=box, x=x, y=y, center=center):
            # The token "box" is stored within the SpanGroup's .box_group
            allocated_tokens.append(token)
        elif token.spans[0].box is not None and token.spans[0].box.is_overlap(other=box, x=x, y=y, center=center):
            # default to assuming the token "box" is stored in the SpanGroup .box
            allocated_tokens.append(token)
        else:
            remaining_tokens.append(token)
    return allocated_tokens, remaining_tokens


def box_groups_to_span_groups(
        box_groups: List[BoxGroup], doc: Document, pad_x: bool = False, center: bool = False
) -> List[SpanGroup]:
    """Generate SpanGroups from BoxGroups.
    Args
        `box_groups` (List[BoxGroup])
        `doc` (Document) base document annotated with pages, tokens, rows to
        `center` (bool) if True, considers tokens to be overlapping with boxes only if their centers overlap
    Returns
        List[SpanGroup] with each SpanGroup.spans corresponding to spans (sans boxes) of allocated tokens per box_group,
        and each SpanGroup.box_group containing original box_groups
    """
    assert all([isinstance(group, BoxGroup) for group in box_groups])

    all_page_tokens = dict()
    avg_token_widths = dict()
    derived_span_groups = []
    token_box_in_box_group = None

    for box_id, box_group in enumerate(box_groups):

        all_tokens_overlapping_box_group = []

        for box in box_group.boxes:

            # Caching the page tokens to avoid duplicated search
            if box.page not in all_page_tokens:
                cur_page_tokens = all_page_tokens[box.page] = doc.pages[
                    box.page
                ].tokens
                if token_box_in_box_group is None:
                    # Determine whether box is stored on token SpanGroup span.box or in the box_group
                    token_box_in_box_group = all(
                        [
                            (
                                    (hasattr(token.box_group, "boxes") and len(token.box_group.boxes) == 1)
                                    and token.spans[0].box is None
                            )
                            for token in cur_page_tokens
                        ]
                    )
                    # Determine average width of tokens on this page if we are going to pad x
                    if pad_x:
                        if token_box_in_box_group and box.page not in avg_token_widths:
                            avg_token_widths[box.page] = np.average([t.box_group.boxes[0].w for t in cur_page_tokens])
                        elif not token_box_in_box_group and box.page not in avg_token_widths:
                            avg_token_widths[box.page] = np.average([t.spans[0].box.w for t in cur_page_tokens])

            else:
                cur_page_tokens = all_page_tokens[box.page]

            # Find all the tokens within the box
            tokens_in_box, remaining_tokens = allocate_overlapping_tokens_for_box(
                tokens=cur_page_tokens,
                box=box,
                token_box_in_box_group=token_box_in_box_group,
                # optionally pad x a small amount so that extra narrow token boxes (when split at punctuation) are not missed
                x=avg_token_widths.get(box.page, 0.0) * 0.5 if pad_x else 0.0,
                y=0.0,
                center=center
            )
            all_page_tokens[box.page] = remaining_tokens

            all_tokens_overlapping_box_group.extend(tokens_in_box)

        merge_spans = (
            MergeSpans.from_span_groups_with_box_groups(
                span_groups=all_tokens_overlapping_box_group, index_distance=1
            )
            if token_box_in_box_group
            else MergeSpans(
                list_of_spans=list(
                    itertools.chain.from_iterable(
                        span_group.spans for span_group in all_tokens_overlapping_box_group
                    )
                ),
                index_distance=1,
            )
        )

        derived_span_groups.append(
            SpanGroup(
                spans=merge_spans.merge_neighbor_spans_by_symbol_distance(),
                box_group=box_group,
                # id = box_id,
            )
            # TODO Right now we cannot assign the box id, or otherwise running doc.blocks will
            # generate blocks out-of-the-specified order.
        )

    if not token_box_in_box_group:
        logging.warning("tokens with box stored in SpanGroup span.box will be deprecated (that is, "
                        "future Spans wont contain box). Ensure Document is annotated with tokens "
                        "having box stored in SpanGroup box_group.boxes")

    del all_page_tokens

    derived_span_groups = sorted(
        derived_span_groups, key=lambda span_group: span_group.start
    )
    # ensure they are ordered based on span indices

    for box_id, span_group in enumerate(derived_span_groups):
        span_group.id = box_id

    # return self._annotate_span_group(
    #     span_groups=derived_span_groups, field_name=field_name
    # )
    return derived_span_groups

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
            [span.start for span in self.list_of_spans]
        )
        ends_matrix = np.full(
            (len(self.list_of_spans), len(self.list_of_spans)),
            [span.end for span in self.list_of_spans]
        )

        starts_minus_ends = np.abs(starts_matrix - ends_matrix.T)
        ends_minus_starts = np.abs(ends_matrix - starts_matrix.T)
        are_neighboring_spans = np.minimum(starts_minus_ends, ends_minus_starts) <= self.index_distance
        neighboring_spans =  np.transpose(are_neighboring_spans.nonzero())

        if len(neighboring_spans) > 0:
            neighboring_spans_no_dupes = neighboring_spans[np.where(neighboring_spans[:,1] < neighboring_spans[:,0])]

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
