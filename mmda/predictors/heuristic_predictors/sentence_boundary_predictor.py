from typing import List, Union, Dict, Any, Tuple

import pysbd
import numpy as np

from mmda.types.annotation import SpanGroup
from mmda.types.span import Span
from mmda.types.document import Document
from mmda.types.names import *
from mmda.predictors.base_predictors.base_heuristic_predictor import (
    BaseHeuristicPredictor,
)


class PysbdSentenceBoundaryPredictor(BaseHeuristicPredictor):
    """Sentence Boundary based on Pysbd

    Examples:
        >>> doc: Document = parser.parse("path/to/pdf")
        >>> predictor = PysbdSentenceBoundaryPredictor()
        >>> sentence_spans = predictor.predict(doc)
        >>> doc.annotate(sentences=sentence_spans)
    """

    REQUIRED_BACKENDS = ["pysbd"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Tokens]

    def __init__(self) -> None:

        self._segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

    def split_token_based_on_sentences_boundary(
        self, words: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Split a list of words into a list of (start, end) indices, indicating the start and end of each sentence.
        Duplicate of https://github.com/allenai/VILA/blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/dataset/preprocessors/layout_indicator.py#L14

        Returns: List[Tuple(int, int)]
            a list of (start, end) for token indices within each sentence
        """

        if len(words) == 0:
            return [(0, 0)]
        combined_words = " ".join(words)

        char2token_mask = np.zeros(len(combined_words), dtype=np.int)

        acc_word_len = 0
        for idx, word in enumerate(words):
            word_len = len(word) + 1
            char2token_mask[acc_word_len : acc_word_len + word_len] = idx
            acc_word_len += word_len

        segmented_sentences = self._segmenter.segment(combined_words)
        sent_boundary = [(ele.start, ele.end) for ele in segmented_sentences]

        split = []
        token_id_start = 0
        for (start, end) in sent_boundary:
            token_id_end = char2token_mask[start:end].max()
            if (
                end + 1 >= len(char2token_mask)
                or char2token_mask[end + 1] != token_id_end
            ):
                token_id_end += 1  # (Including the end)
            split.append((token_id_start, token_id_end))
            token_id_start = token_id_end
        return split

    def predict(self, doc: Document) -> List[SpanGroup]:

        words = [token.symbols[0] for token in doc.tokens]
        split = self.split_token_based_on_sentences_boundary(words)

        sentence_spans = []
        for (start, end) in split:
            if end - start == 0:
                continue
            if end - start < 0:
                raise ValueError
            cur_spans = doc.tokens[start:end]
            start = min([ele.start for ele in cur_spans])
            end = max([ele.end for ele in cur_spans])
            sentence_spans.append(SpanGroup([Span(start, end)]))

        return sentence_spans