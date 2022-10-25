"""

Dataclass for representing a document and all its constituents

@kylel

"""

from typing import List, Optional, Dict, Tuple, Type

from intervaltree import IntervalTree

from mmda.types.boundingbox import BoundingBox
from mmda.types.annotations import Annotation, SpanAnnotation, BoundingBoxAnnotation
from mmda.types.image import Image
from mmda.types.span import Span


Text = 'text'
Page = 'page'
Token = 'token'
Row = 'row'
Sent = 'sent'
Block = 'block'
DocImage = 'image' # Conflicting the PIL Image naming 


class Document:

    valid_types = [Page, Token, Row, Sent, Block]

    def __init__(self, text: str):

        self.text = text

        # TODO: if have span_type Map, do still need these?
        self._pages: List[Span] = []
        self._tokens: List[Span] = []
        self._rows: List[Span] = []
        self._sents: List[Span] = []
        self._blocks: List[Span] = []
        self._images: List["PIL.Image"] = []

        self._span_type_to_spans: Dict[Type, List[Span]] = {
            Page: self._pages,
            Token: self._tokens,
            Row: self._rows,
            Sent: self._sents,
            Block: self._blocks
        }

        self._page_index: IntervalTree = IntervalTree()
        self._token_index: IntervalTree = IntervalTree()
        self._row_index: IntervalTree = IntervalTree()
        self._sent_index: IntervalTree = IntervalTree()
        self._block_index: IntervalTree = IntervalTree()

        self._span_type_to_index: Dict[Type, IntervalTree] = {
            Page: self._page_index,
            Token: self._token_index,
            Row: self._row_index,
            Sent: self._sent_index,
            Block: self._block_index
        }

    @classmethod
    def from_json(cls, doc_json: Dict) -> 'Document':
        doc = Document(text=doc_json[Text])
        pages = []
        tokens = []
        rows = []
        sents = []
        blocks = []

        for span_type in cls.valid_types:
            if span_type in doc_json:
                doc_spans = [DocSpan.from_span(span=Span.from_json(span_json=span_json), doc=doc, span_type=span_type)
                             for span_json in doc_json[span_type]]
                if span_type == Page:
                    pages = doc_spans
                elif span_type == Token:
                    tokens = doc_spans
                elif span_type == Row:
                    rows = doc_spans
                elif span_type == Sent:
                    sents = doc_spans
                elif span_type == Block:
                    blocks = doc_spans
                else:
                    raise Exception(f'Should never reach here')
        
        images = [Image.frombase64(image_str) for image_str in doc_json.get(DocImage,[])]

        doc.load(pages=pages, tokens=tokens, rows=rows, sents=sents, blocks=blocks, images=images)
        return doc

    # TODO: consider simpler more efficient method (e.g. JSONL; text)
    def to_json(self) -> Dict:
        return {
            Text: self.text,
            Page: [page.to_json(exclude=['text', 'type']) for page in self.pages],
            Token: [token.to_json(exclude=['text', 'type']) for token in self.tokens],
            Row: [row.to_json(exclude=['text', 'type']) for row in self.rows],
            Sent: [sent.to_json(exclude=['text', 'type']) for sent in self.sents],
            Block: [block.to_json(exclude=['text', 'type']) for block in self.blocks],
            DocImage: [image.tobase64() for image in self.images]
        }

    #
    #   methods for building Document
    #
    def _build_span_index(self, spans: List[Span]) -> IntervalTree:
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
        self._span_type_to_spans: Dict[Type, List[Span]] = {
            Page: self._pages,
            Token: self._tokens,
            Row: self._rows,
            Sent: self._sents,
            Block: self._blocks
        }

    def _build_span_type_to_index(self):
        self._span_type_to_index: Dict[Type, IntervalTree] = {
            Page: self._page_index,
            Token: self._token_index,
            Row: self._row_index,
            Sent: self._sent_index,
            Block: self._block_index
        }

    def load(self, pages: Optional[List[Span]] = None,
             tokens: Optional[List[Span]] = None,
             rows: Optional[List[Span]] = None,
             sents: Optional[List[Span]] = None,
             blocks: Optional[List[Span]] = None,
             images: Optional[List["PIL.Image"]] = None):

        if pages:
            self._pages = pages
            self._page_index = self._build_span_index(spans=pages)
        if tokens:
            self._tokens = tokens
            self._token_index = self._build_span_index(spans=tokens)
        if rows:
            self._rows = rows
            self._row_index = self._build_span_index(spans=rows)
        if sents:
            self._sents = sents
            self._sent_index = self._build_span_index(spans=sents)
        if blocks:
            self._blocks = blocks
            self._block_index = self._build_span_index(spans=blocks)
        if images:
            self._images = images

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

    @property
    def images(self) -> List["PIL.Image"]:
        return self._images

    #
    #   methods for using Document
    #
    # TODO: how should `containment` lookups be handled in this library? intersection is symmetric but containment isnt
    # TODO: @lru.cache or some memoization might improve performance
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
