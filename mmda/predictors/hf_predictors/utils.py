from mmda.types.annotation import SpanGroup
from typing import List, Tuple
import itertools

from mmda.types.document import Document
from mmda.types.names import *


def shift_index_sequence_to_zero_start(sequence):
    """
    Shift a sequence to start at 0.
    """
    sequence_start = min(sequence)
    return [i - sequence_start for i in sequence]


def get_visual_group_id(token: SpanGroup, field_name: str, defaults=-1) -> int:

    field_value = getattr(token, field_name)
    if len(field_value) == 0 or field_value[0].id is None:
        return defaults
    return field_value[0].id


def convert_document_page_to_pdf_dict(document: Document):
    """
    Convert a document to a dictionary of the form:
    {
        'words': ['word1', 'word2', ...],
        'bbox': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
        'block_ids': [0, 0, 0, 1 ...],
        'line_ids': [0, 1, 1, 2 ...],
        'labels': [0, 0, 0, 1 ...], # could be empty
    }
    """

    words = [token.symbols[0] for token in document.tokens]
    # TODO: Right now we assume the token could on have a single span.

    bbox = [token.spans[0].box.coordinates for token in document.tokens]
    # TODO: This returns relative coordinates to the document.

    line_ids = [get_visual_group_id(token, Rows, -1) for token in document.tokens]
    # TODO: Right now we assume the token could span for one row of the
    # document.
    line_ids = shift_index_sequence_to_zero_start(line_ids)

    block_ids = [get_visual_group_id(token, Blocks, -1) for token in document.tokens]
    # TODO: Right now we assume the token could span for one block of the
    # document.
    block_ids = shift_index_sequence_to_zero_start(block_ids)

    labels = [None] * len(words)
    # TODO: We provide an empty label list.

    return {
        "words": words,
        "bbox": bbox,
        "block_ids": block_ids,
        "line_ids": line_ids,
        "labels": labels,
    }


def convert_sequence_tagging_to_spans(
    token_prediction_sequence: List,
) -> List[Tuple[int, int, int]]:
    """For a squence of token predictions, convert them to spans
    of consecutive same predictions.

    Args:
        token_prediction_sequence (List)

    Returns:
        List[Tuple[int, int, int]]: A list of (start, end, label)
            of consecutive prediction of the same label.
    """
    prev_len = 0
    spans = []
    for gp, seq in itertools.groupby(token_prediction_sequence):
        cur_len = len(list(seq))
        spans.append((prev_len, prev_len + cur_len, gp))
        prev_len = prev_len + cur_len
    return spans