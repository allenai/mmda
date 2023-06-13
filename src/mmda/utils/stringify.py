"""

Utility that converts a List[SpanGroup] into a string

@kylel, @lucas

Prior versions: 
- https://github.com/allenai/timo_scim/blob/9fa19cd29cde0e2573e0079da8d776895f4d6caa/scim/predictors/utils.py#L221-L234
- 

"""

import logging
from typing import List, Optional

from mmda.types import Document, Span, SpanGroup

# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def stringify_span_group(
    span_group: SpanGroup,
    document: Document,
    use_word_text: bool = True,
    replace_newlines_with: str = " ",
    join_words_with: str = " ",
    normalize_whitespace: bool = True,
    allow_partial_word_match: bool = True,
    allow_disjoint_spans: bool = True,
    include_symbols_between_disjoint_spans: bool = False,
) -> str:
    """Requires doc.words to exist. This technique applies 4 main steps:

    1. Given a query span_group, find associated Words
    2. Given associated Words, find associated Symbols (i.e. start-end indices)
    3. Given associated Symbols, produce List[str] which includes whitespace chars
    4. Given List[str], join into a single string
    """
    matched_words = document.find_overlapping(span_group, "words")

    # warn if no words intersecting query span_group
    if len(matched_words) == 0:
        logger.debug(f"span_group {span_group} has no overlapping words in document")
        return ""

    # are words cut in half? suppose word is (0, 3), but the query asks for (2, 3). Should it include the whole word?
    match_any_word_start = any(
        [word.start == span_group.start for word in matched_words]
    )
    match_any_word_end = any([word.end == span_group.end for word in matched_words])
    if allow_partial_word_match:
        if not match_any_word_start:
            logger.debug(
                f"span_group {span_group}'s start index doesnt match w/ start of any word. output string may thus include more text than expected."
            )
        if not match_any_word_end:
            logger.debug(
                f"span_group {span_group}'s end index doesnt match w/ end of any word. output string may thus include more text than expected."
            )
    else:
        if not match_any_word_start:
            raise ValueError(
                f"span_group {span_group}'s start index doesnt match w/ start of any word"
            )
        if not match_any_word_end:
            raise ValueError(
                f"span_group {span_group}'s end index doesnt match w/ end of any word"
            )

    # if query has disjoint spans, what should we do with the in-between symbols when stringifying?
    if Span.are_disjoint(spans=span_group.spans):
        if not allow_disjoint_spans:
            raise ValueError(
                f"span_group {span_group} has disjoint spans but allow_disjoint_spans is False"
            )
        else:
            logger.debug(f"span_group {span_group} has disjoint spans")
            if include_symbols_between_disjoint_spans:
                # TODO: easiest is probably to convert disjoint spans into a single longer span
                raise NotImplementedError

    # if matched words are disjoint, what should we do with the in-between symbols when stringifying?
    if Span.are_disjoint(spans=[span for word in matched_words for span in word.spans]):
        if not allow_disjoint_spans:
            raise ValueError(
                f"span_group {span_group} intersects words {matched_words} which have disjoint spans but allow_disjoint_spans is False"
            )
        else:
            logger.debug(
                f"span_group {span_group} intersects words {matched_words} which have disjoint spans"
            )
            if include_symbols_between_disjoint_spans:
                # TODO: easiest is probably to convert disjoint spans into a single longer span
                raise NotImplementedError

    # TODO: actually, maybe refactor this. it doesnt matter if query spangroup is disjoint
    #       the actual handling should just happen if it's disjoint words.

    # define util function that produces text from a word
    _stringify_word = lambda word: word.text if use_word_text else None
    if use_word_text is False:
        raise NotImplementedError(
            f"""Havnt figured out how to do this yet. Only supports use_word_text=True for now.
              One possible consideration is `document.symbols[start:end]` as a means to get
              text, which may be useful for debugging purposes. But this isn't high priority 
              right now."""
        )

    # define util function that replaces all whitespace with a single character
    _normalize_whitespace = lambda text: " ".join(text.split())

    # stringify!
    prev = matched_words[0]
    text_from_words_including_whitespace: List[str] = [_stringify_word(word=prev)]
    for current in matched_words[1:]:
        is_there_a_gap = current.start > prev.end
        if is_there_a_gap:
            text_from_words_including_whitespace.append(join_words_with)
        text_from_words_including_whitespace.append(_stringify_word(word=current))
        prev = current

    # final step of formatting the text to return
    candidate_text = "".join(text_from_words_including_whitespace)
    if replace_newlines_with is not None:
        candidate_text = candidate_text.replace("\n", replace_newlines_with)
    if normalize_whitespace:
        candidate_text = _normalize_whitespace(candidate_text)
    return candidate_text
