"""

Dataclass for representing a document and all its constituents

@kylel

"""

from typing import List, Optional, Dict

from mmda.types.annotations import Annotation, SpanAnnotation, BoundingBoxAnnotation
from mmda.types.span import Span


class Document:

    def __init__(self, text: str):
        self.text = text
        self.pages: Optional[List[Span]] = None
        self.tokens: Optional[List[Span]] = None
        self.rows: Optional[List[Span]] = None
        self.sents: Optional[List[Span]] = None
        self.blocks: Optional[List[Span]] = None

    @classmethod
    def from_json(cls, doc_json: Dict) -> 'Document':
        doc = Document(text=doc_json['text'])
        if 'pages' in doc_json:
            pages = [Span.from_json(span_json=page_json) for page_json in doc_json['pages']]
            doc.load_pages(pages=pages)
        if 'tokens' in doc_json:
            tokens = [Span.from_json(span_json=token_json) for token_json in doc_json['tokens']]
            doc.load_tokens(tokens=tokens)
        if 'rows' in doc_json:
            rows = [Span.from_json(span_json=row_json) for row_json in doc_json['rows']]
            doc.load_rows(rows=rows)
        if 'sents' in doc_json:
            tokens = [Span.from_json(span_json=token_json) for token_json in doc_json['tokens']]
            doc.load_tokens(tokens=tokens)
        if 'blocks' in doc_json:
            blocks = [Span.from_json(span_json=block_json) for block_json in doc_json['blocks']]
            doc.load_blocks(blocks=blocks)
        return doc

    def to_json(self):
        raise NotImplementedError

    def load_pages(self, pages: List[Span]):
        self.pages = pages

    def load_tokens(self, tokens: List[Span]):
        self.tokens = tokens

    def load_rows(self, rows: List[Span]):
        self.rows = rows

    def load_sents(self, sents: List[Span]):
        self.sents = sents

    def load_blocks(self, blocks: List[Span]):
        self.blocks = blocks


    # TODO - gennerally, we care about relations between pairs of Span (e.g. token-sent, token-row, sent-page, sent-block, row-block);  containment & intersection seem primary

    # TODO - how do we enforce constraints on these (e.g. token-sent should adhere to boundaries; other strict nesting relations)

    # TODO - how do we build index for fast lookup between spans (e.g. sents.token should work;  token.sent should work)

    # TODO - how to build these up;  ordering?  allowable None patterns?

    # TODO: what happens to the document data when annotate?
    def annotate(self, annotations: List[Annotation]):
        for annotation in annotations:
            if isinstance(annotation, SpanAnnotation):
                pass
            elif isinstance(annotation, BoundingBoxAnnotation):
                pass
            else:
                pass
        raise NotImplementedError

