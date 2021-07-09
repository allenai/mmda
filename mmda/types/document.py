"""

Dataclass for representing a document and all its constituents

@kylel

"""

from typing import List, Optional, Dict, Tuple, Type

from intervaltree import IntervalTree

from mmda.types.boundingbox import BoundingBox
from mmda.types.annotations import Annotation, SpanAnnotation, BoundingBoxAnnotation
from mmda.types.span import Span


class Document:

    valid_types = ['page', 'token', 'row', 'sent', 'block']

    def __init__(self, text: str):
        self.text = text

        self._pages: Optional[List[Span]] = None
        self._tokens: Optional[List[Span]] = None
        self._rows: Optional[List[Span]] = None
        self._sents: Optional[List[Span]] = None
        self._blocks: Optional[List[Span]] = None

        self._span_type_to_spans: Dict[Type, Optional[Span]] = {
            'page': self._pages,
            'token': self._tokens,
            'row': self._rows,
            'sent': self._sents,
            'block': self._blocks
        }

        self._index_pages: Optional[IntervalTree] = None
        self._index_tokens: Optional[IntervalTree] = None
        self._index_rows: Optional[IntervalTree] = None
        self._index_sents: Optional[IntervalTree] = None
        self._index_blocks: Optional[IntervalTree] = None

        self._span_type_to_index: Dict[Type, Optional[IntervalTree]] = {
            'page': self._index_pages,
            'token': self._index_tokens,
            'row': self._index_rows,
            'sent': self._index_sents,
            'block': self._index_blocks
        }

    @classmethod
    def from_json(cls, doc_json: Dict) -> 'Document':
        doc = Document(text=doc_json['text'])
        for span_type in cls.valid_types:
            if span_type in doc_json:
                doc_spans = [DocSpan.from_span(span=Span.from_json(span_json=span_json), doc=doc, span_type=span_type)
                             for span_json in doc_json[span_type]]
                if span_type == 'page':
                    doc.load_pages(pages=doc_spans)
                elif span_type == 'token':
                    doc.load_tokens(tokens=doc_spans)
                elif span_type == 'row':
                    doc.load_rows(rows=doc_spans)
                elif span_type == 'sent':
                    doc.load_sents(sents=doc_spans)
                elif span_type == 'block':
                    doc.load_blocks(blocks=doc_spans)
                else:
                    raise Exception(f'Should never reach here')
        return doc

    def to_json(self) -> Dict:
        return {
            'text': self.text,
            'page': [page.to_json(exclude=['text', 'type']) for page in self.pages],
            'token': [token.to_json(exclude=['text', 'type']) for token in self.tokens],
            'row': [row.to_json(exclude=['text', 'type']) for row in self.rows],
            'sent': [sent.to_json(exclude=['text', 'type']) for sent in self.sents],
            'block': [block.to_json(exclude=['text', 'type']) for block in self.blocks]
        }

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
            'page': self._pages,
            'token': self._tokens,
            'row': self._rows,
            'sent': self._sents,
            'block': self._blocks
        }

    def _build_span_type_to_index(self):
        self._span_type_to_index: Dict[Type, Optional[IntervalTree]] = {
            'page': self._index_pages,
            'token': self._index_tokens,
            'row': self._index_rows,
            'sent': self._index_sents,
            'block': self._index_blocks
        }

    def load_pages(self, pages: List[Span]):
        self._pages = pages
        self._index_pages = self._index_spans(spans=pages)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_tokens(self, tokens: List[Span]):
        self._tokens = tokens
        self._index_tokens = self._index_spans(spans=tokens)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_rows(self, rows: List[Span]):
        self._rows = rows
        self._index_rows = self._index_spans(spans=rows)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_sents(self, sents: List[Span]):
        self._sents = sents
        self._index_sents = self._index_spans(spans=sents)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    def load_blocks(self, blocks: List[Span]):
        self._blocks = blocks
        self._index_blocks = self._index_spans(spans=blocks)
        self._build_span_type_to_spans()
        self._build_span_type_to_index()

    #
    #   don't mess with Document internals
    #
    @property
    def pages(self) -> List[Span]:
        return self._pages

    @property
    def tokens(self) -> List[Span]:
        return self._tokens

    @property
    def rows(self) -> List[Span]:
        return self._rows

    @property
    def sents(self) -> List[Span]:
        return self._sents

    @property
    def blocks(self) -> List[Span]:
        return self._blocks

    #
    #   methods for using Document
    #
    # TODO: how should `containment` lookups be handled in this library? intersection is symmetric but containment isnt
    # TODO: @lru.cache or some memoization might improve performance
    # TODO: cleaner interface would be to make this private, make Token/Page/Block/Sentence/Row aware of Document & use `.<classname>` interface
    def find(self, query: Span, types: str) -> List[Span]:
        index = self._span_type_to_index[types]
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



class DocSpan(Span):
    def __init__(self, start: int, end: int, doc: Document,
                 id: Optional[int] = None, type: Optional[str] = None,
                 text: Optional[str] = None, bbox: Optional[BoundingBox] = None):
        super().__init__(start=start, end=end, id=id, type=type, text=text, bbox=bbox)
        self.doc = doc

    @property
    def tokens(self) -> List:
        if self.type == 'token':
            raise ValueError(f'{self} is a Token and cant lookup other Tokens')
        else:
            return self.doc.find(query=self, types='token')

    @property
    def pages(self) -> List:
        if self.type == 'page':
            raise ValueError(f'{self} is a Page and cant lookup other Pages')
        else:
            return self.doc.find(query=self, types='page')

    @property
    def rows(self) -> List:
        if self.type == 'row':
            raise ValueError(f'{self} is a Row and cant lookup other Rows')
        else:
            return self.doc.find(query=self, types='row')

    @property
    def sents(self) -> List:
        if self.type == 'sent':
            raise ValueError(f'{self} is a Sentence and cant lookup other Sentences')
        else:
            return self.doc.find(query=self, types='sent')

    @property
    def blocks(self) -> List:
        if self.type == 'block':
            raise ValueError(f'{self} is a Block and cant lookup other Blocks')
        else:
            return self.doc.find(query=self, types='block')

    @classmethod
    def from_span(cls, span: Span, doc: Document, span_type: str) -> 'DocSpan':
        doc_span = cls(start=span.start, end=span.end, doc=doc,
                       type=span.type, id=span.id, text=span.text, bbox=span.bbox)
        # these two fields are optional for `Span` & not often serialized in span_jsons, but are
        # critical for DocSpan methods to work properly
        if not doc_span.type:
            doc_span.type = span_type
        if not doc_span.text:
            doc_span.text = doc.text[doc_span.start:doc_span.end]
        return doc_span
