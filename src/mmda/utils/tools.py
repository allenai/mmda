from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np

from mmda.types import BoxGroup, Document, Span, SpanGroup


def allocate_overlapping_tokens_for_box(
    tokens: List[SpanGroup],
    box,
    token_box_in_box_group: bool = False,
    x: float = 0.0,
    y: float = 0.0,
    center: bool = False,
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
        if token_box_in_box_group and token.box_group.boxes[0].is_overlap(
            other=box, x=x, y=y, center=center
        ):
            # The token "box" is stored within the SpanGroup's .box_group
            allocated_tokens.append(token)
        elif token.spans[0].box is not None and token.spans[0].box.is_overlap(
            other=box, x=x, y=y, center=center
        ):
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
                cur_page_tokens = all_page_tokens[box.page] = doc.pages[box.page].tokens
                if token_box_in_box_group is None:
                    # Determine whether box is stored on token SpanGroup span.box or in the box_group
                    token_box_in_box_group = all(
                        [
                            (
                                (
                                    hasattr(token.box_group, "boxes")
                                    and len(token.box_group.boxes) == 1
                                )
                                and token.spans[0].box is None
                            )
                            for token in cur_page_tokens
                        ]
                    )
                    # Determine average width of tokens on this page if we are going to pad x
                    if pad_x:
                        if token_box_in_box_group and box.page not in avg_token_widths:
                            avg_token_widths[box.page] = np.average(
                                [t.box_group.boxes[0].w for t in cur_page_tokens]
                            )
                        elif (
                            not token_box_in_box_group
                            and box.page not in avg_token_widths
                        ):
                            avg_token_widths[box.page] = np.average(
                                [t.spans[0].box.w for t in cur_page_tokens]
                            )

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
                center=center,
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
                        span_group.spans
                        for span_group in all_tokens_overlapping_box_group
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
        logging.warning(
            "tokens with box stored in SpanGroup span.box will be deprecated (that is, "
            "future Spans wont contain box). Ensure Document is annotated with tokens "
            "having box stored in SpanGroup box_group.boxes"
        )

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
