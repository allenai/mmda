from typing import Iterable

import mmda.types.annotation as legacy
from mmda.predictors.hf_predictors.vila_predictor import HVILAPredictor
from mmda.types.nouveau.base import Document, Span
from mmda.types.nouveau.protocol import SpanPredictor
from mmda.types.nouveau.scientific import TokenGroup, TokenType

# This is not a perfect mapping as some GROTOAP2 categories correspond to multiple S2-VL
# mappings. So, some possible information is lost in this translation.
GROTOAP2_TOKEN_TYPE_MAPPING = {
    # BIB_INFO
    0: TokenType.Footer,
    # REFERENCE
    1: TokenType.Bibliography,
    # UNKNOWN
    2: TokenType.Unknown,
    # BODY_CONTENT
    3: TokenType.Paragraph,
    # PAGE_NUMBER
    4: TokenType.Paragraph,
    # TABLE
    5: TokenType.Table,
    # ACKNOWLEDGEMENT
    6: TokenType.Paragraph,
    # FIGURE
    7: TokenType.Caption,
    # CONFLICT_STATEMENT
    8: TokenType.Paragraph,
    # AFFILIATION
    9: TokenType.Author,
    # DATES
    10: TokenType.Paragraph,
    # TYPE
    11: TokenType.Paragraph,
    # TITLE
    12: TokenType.Title,
    # AUTHOR
    13: TokenType.Author,
    # ABSTRACT
    14: TokenType.Abstract,
    # CORRESPONDENCE
    15: TokenType.Paragraph,
    # EDITOR
    16: TokenType.Paragraph,
    # COPYRIGHT
    17: TokenType.Paragraph,
    # AUTHOR_TITLE
    18: TokenType.Author,
    # KEYWORDS
    19: TokenType.Keywords,
    # GLOSSARY
    20: TokenType.Paragraph,
    # EQUATION
    21: TokenType.Equation,
}


class NouveauHVilaPredictor(SpanPredictor):
    model: HVILAPredictor

    def __init__(self, model: HVILAPredictor) -> None:
        self.model = model

    @classmethod
    def from_pretrained(cls, model_name: str, **preprocessor_config):
        return NouveauHVilaPredictor(
            HVILAPredictor.from_pretrained(model_name, **preprocessor_config)
        )

    def predict(self, document: Document) -> Iterable[TokenGroup]:
        preds = self.model.predict(document)
        import pdb

        pdb.set_trace()
        return [self._convert_span_group(sg) for sg in self.model.predict(document)]

    def _convert_span_group(self, legacy_span_group: legacy.SpanGroup) -> TokenGroup:
        # FIXME: Here legacy_span.box is not required but we want it!
        spans = [
            Span(start=legacy_span.start, end=legacy_span.end, box=legacy_span.box)
            for legacy_span in legacy_span_group.spans
        ]
        token_group = TokenGroup(spans=spans)
        token_group.type = self._map_legacy_type_to_token_type(legacy_span_group.type)

        return token_group

    # FIXME: This will assume GROTOAP2 for now
    def _map_legacy_type_to_token_type(self, token_type: int) -> TokenType:
        return GROTOAP2_TOKEN_TYPE_MAPPING[token_type]
