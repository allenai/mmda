"""

Uses a whitespace tokenizer on Document.symbols to predict which `tokens` were originally
part of the same segment/chunk (e.g. "few-shot" if tokenized as ["few", "-", "shot"]).

@kylel

"""

from typing import Optional, Set, List, Tuple

import tokenizers
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types import Metadata, Document, SpanGroup, Span, BoxGroup
from mmda.types.names import TokensField


class WhitespacePredictor(BasePredictor):
    REQUIRED_BACKENDS = None
    REQUIRED_DOCUMENT_FIELDS = [TokensField]

    _dictionary: Optional[Set[str]] = None

    def __init__(self) -> None:
        self.whitespace_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    def predict(self, document: Document) -> List[SpanGroup]:
        self._doc_field_checker(document)

        # 1) whitespace tokenization on symbols. each token is a nested tuple ('text', (start, end))
        ws_tokens: List[Tuple] = self.whitespace_tokenizer.pre_tokenize_str(document.symbols)

        # 2) filter to just the chunks that are greater than 1 token. Reformat.
        # chunks = []
        # for text, (start, end) in ws_tokens:
        #     overlapping_tokens = document.find_overlapping(
        #         query=SpanGroup(spans=[Span(start=start, end=end)]),
        #         field_name=Tokens
        #     )
        #     if len(overlapping_tokens) > 1:
        #         chunk = SpanGroup(spans=[Span(start=start, end=end)], metadata=Metadata(text=text))
        #         chunks.append(chunk)
        chunks = []
        for i, (text, (start, end)) in enumerate(ws_tokens):
            chunk = SpanGroup(spans=[Span(start=start, end=end)],
                              metadata=Metadata(text=text),
                              id=i)
            chunks.append(chunk)
        return chunks
