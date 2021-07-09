"""

Dataclass for representing a document and all its constituents

@kylel

"""

from typing import List, Optional, Dict, Tuple, Type

from intervaltree import IntervalTree

from mmda.types.boundingbox import BoundingBox
from mmda.types.annotations import Annotation, SpanAnnotation, BoundingBoxAnnotation
from mmda.types.span import Span


class DocumentSpan(Span):
    def __init__(self, start: int, end: int, doc: Document, id: Optional[int] = None,
                 text: Optional[str] = None, bbox: Optional[BoundingBox] = None):
        super().__init__(start=start, end=end, id=id, text=text, bbox=bbox)
        self.doc = doc

    @property
    def tokens(self) -> List[Token]:
        if type(self) == Token:
            raise ValueError(f'{self} is a Token and cant lookup other Tokens')
        else:
            return self.doc.find(query=self, type=Token)

    @property
    def pages(self) -> List[Page]:
        if type(self) == Page:
            raise ValueError(f'{self} is a Page and cant lookup other Pages')
        else:
            return self.doc.find(query=self, type=Page)

    @property
    def rows(self) -> List[Row]:
        if type(self) == Row:
            raise ValueError(f'{self} is a Row and cant lookup other Rows')
        else:
            return self.doc.find(query=self, type=Row)

    @property
    def sents(self) -> List[Sentence]:
        if type(self) == Sentence:
            raise ValueError(f'{self} is a Sentence and cant lookup other Sentences')
        else:
            return self.doc.find(query=self, type=Sentence)

    @property
    def blocks(self) -> List[Block]:
        if type(self) == Block:
            raise ValueError(f'{self} is a Block and cant lookup other Blocks')
        else:
            return self.doc.find(query=self, type=Block)


class Token(DocumentSpan):
    pass

class Page(DocumentSpan):
    pass

class Row(DocumentSpan):
    pass

class Sentence(DocumentSpan):
    pass

class Block(DocumentSpan):
    pass


class Document:
    def __init__(self, text: str):
        self.text = text

        self._pages: Optional[List[Page]] = None
        self._tokens: Optional[List[Token]] = None
        self._rows: Optional[List[Row]] = None
        self._sents: Optional[List[Sentence]] = None
        self._blocks: Optional[List[Block]] = None

        self._span_type_to_spans: Dict[Type, Optional[Span]] = {
            Page: self._pages,
            Token: self._tokens,
            Row: self._rows,
            Sentence: self._sents,
            Block: self._blocks
        }

        self._index_pages: Optional[IntervalTree] = None
        self._index_tokens: Optional[IntervalTree] = None
        self._index_rows: Optional[IntervalTree] = None
        self._index_sents: Optional[IntervalTree] = None
        self._index_blocks: Optional[IntervalTree] = None

        self._span_type_to_index: Dict[Type, Optional[IntervalTree]] = {
            Page: self._index_pages,
            Token: self._index_tokens,
            Row: self._index_rows,
            Sentence: self._index_sents,
            Block: self._index_blocks
        }

    @classmethod
    def from_json(cls, doc_json: Dict) -> 'Document':
        doc = Document(text=doc_json['text'])
        if 'pages' in doc_json:
            pages = [Page.from_json(span_json=page_json) for page_json in doc_json['pages']]
            doc.load_pages(pages=pages)
        if 'tokens' in doc_json:
            tokens = [Token.from_json(span_json=token_json) for token_json in doc_json['tokens']]
            doc.load_tokens(tokens=tokens)
        if 'rows' in doc_json:
            rows = [Row.from_json(span_json=row_json) for row_json in doc_json['rows']]
            doc.load_rows(rows=rows)
        if 'sents' in doc_json:
            tokens = [Sentence.from_json(span_json=token_json) for token_json in doc_json['tokens']]
            doc.load_tokens(tokens=tokens)
        if 'blocks' in doc_json:
            blocks = [Block.from_json(span_json=block_json) for block_json in doc_json['blocks']]
            doc.load_blocks(blocks=blocks)
        return doc

    def to_json(self):
        raise NotImplementedError

    #
    #   methods for building Document
    #
    def _index_spans(self, spans: List[Span]) -> IntervalTree:
        """Builds index for a collection of spans"""
        index = IntervalTree()
        for span in spans:
            # constraint - all spans disjoint
            existing = index[span.start:span.end]
            if existing:
                raise ValueError(f'Existing {existing} when attempting index {span}')
            # add to index
            index[span.start:span.end] = span
        return index

    def _build_span_type_to_spans(self):
        self._span_type_to_spans: Dict[Type, Optional[Span]] = {
            Page: self._pages,
            Token: self._tokens,
            Row: self._rows,
            Sentence: self._sents,
            Block: self._blocks
        }

    def _build_span_type_to_index(self):
        self._span_type_to_index: Dict[Type, Optional[IntervalTree]] = {
            Page: self._index_pages,
            Token: self._index_tokens,
            Row: self._index_rows,
            Sentence: self._index_sents,
            Block: self._index_blocks
        }

    def load_pages(self, pages: List[Page]):
        self._pages = pages
        self._index_pages = self._index_spans(spans=pages)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_tokens(self, tokens: List[Token]):
        self._tokens = tokens
        self._index_tokens = self._index_spans(spans=tokens)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_rows(self, rows: List[Row]):
        self._rows = rows
        self._index_rows = self._index_spans(spans=rows)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_sents(self, sents: List[Sentence]):
        self._sents = sents
        self._index_sents = self._index_spans(spans=sents)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_blocks(self, blocks: List[Block]):
        self._blocks = blocks
        self._index_blocks = self._index_spans(spans=blocks)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    #
    #   don't mess with Document internals
    #
    @property
    def pages(self) -> List[Page]:
        return self._pages

    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    @property
    def rows(self) -> List[Row]:
        return self._rows

    @property
    def sents(self) -> List[Sentence]:
        return self._sents

    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    #
    #   methods for using Document
    #
    # TODO: how should `containment` lookups be handled in this library? intersection is symmetric but containment isnt
    # TODO: @lru.cache or some memoization might improve performance
    # TODO: cleaner interface would be to make this private, make Token/Page/Block/Sentence/Row aware of Document & use `.<classname>` interface
    def find(self, query: Span, type: Type) -> List[Span]:
        index = self._span_type_to_index[type]
        return sorted([interval.data for interval in index[query.start:query.end]])

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

