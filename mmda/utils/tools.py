from typing import List, Union, Dict, Any, Tuple, Optional

from mmda.types.span import Span


def merge_neighbor_spans(spans: List[Span], distance: Optional[int]) -> List[Span]:
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