"""

Predict words from candidate tokens

"""

from typing import List, Tuple, Dict, Optional

from mmda.types.document import Document
from mmda.types.annotation import Annotation, SpanGroup
from mmda.types.names import *
from mmda.predictors.base_predictors.base_heuristic_predictor import BaseHeuristicPredictor

class WordHeuristicPredictor(BaseHeuristicPredictor):

    REQUIRED_BACKENDS = []
    REQUIRED_DOCUMENT_FIELDS = [Tokens, Rows, Pages]

    def predict(self, document: Document) -> List[Annotation]:
        """For all the mmda models, the input is a document object, and
        the output is a list of annotations.
        """
        self._doc_field_checker(document)


    def _make_candidate(self, this_row: SpanGroup, next_row: SpanGroup, doc: Document) -> Optional[Dict]:
        this_row_last_token = this_row.tokens[-1]
        assert len(this_row_last_token.spans) == 1, "Token SpanGroups should have a single Span"
        if this_row_last_token.text:
            this_row_last_token_text = this_row_last_token.text
        else:
            this_row_last_token_span = this_row_last_token.spans[0]
            this_row_last_token_text = doc.symbols[this_row_last_token_span.start: this_row_last_token_span.end]
        if this_row_last_token_text.endswith('-'):
            next_row_first_token = next_row.tokens[0]
            assert len(next_row_first_token.spans) == 1, "Token SpanGroups should have a single Span"
            if next_row_first_token.text:
                next_row_first_token_text = next_row_first_token.text
            else:
                next_row_first_token_span = next_row_first_token.spans[0]
                next_row_first_token_text = doc.symbols[next_row_first_token_span.start: next_row_first_token_span.end]
            return {
                'this_row_last_token': this_row_last_token,
                'next_row_first_token': next_row_first_token,
                'this_row_last_token_text': this_row_last_token_text,
                'next_row_first_token_text': next_row_first_token_text
            }
        return None

    def get_candidate_tokens(self, doc: Document) -> List[Dict]:
        candidates = []
        for i in range(len(doc.pages) - 1):
            this_page = doc.pages[i]
            next_page = doc.pages[i + 1]
            for j in range(len(this_page.rows) - 1):
                candidate = self._make_candidate(this_row=this_page.rows[j],
                                                 next_row=this_page.rows[j + 1],
                                                 doc=doc)
                if candidate:
                    candidates.append(candidate)
            # last row on page
            candidate = self._make_candidate(this_row=this_page.rows[-1],
                                             next_row=next_page.rows[0],
                                             doc=doc)
            if candidate:
                candidates.append(candidate)
        return candidates