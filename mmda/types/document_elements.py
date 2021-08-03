from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field
import base64
from io import BytesIO

from intervaltree import IntervalTree
from PIL import Image


@dataclass
class DocumentElement:
    """DocumentElement is the base class for all children objects
    of a Document. It defines the necessary APIs for manipulating
    the children objects.
    """

    @abstractmethod
    def to_json(self) -> Dict:
        pass


@dataclass
class DocumentPageSymbols(DocumentElement):
    """Storing the symbols of a page."""

    symbols: str

    # TODO: Add support for symbol bounding boxes and style
    def __getitem__(self, key):
        return self.symbols[key]

    def to_json(self):
        return self.symbols


@dataclass
class DocumentSymbols(DocumentElement):
    """Storing the symbols of a document."""

    page_count: int
    page_symbols: List[DocumentPageSymbols] = field(default_factory=list)

    def __getitem__(self, indices):
        page_id, symbol_slices = indices
        assert page_id < len(self.page_symbols), "Page index out of range"
        return self.page_symbols[page_id][symbol_slices]

    def to_json(self):
        return [page_symbols.to_json() for page_symbols in self.page_symbols]


@dataclass
class Span:
    start: int
    end: int
    page: int

    def to_json(self) -> Dict:
        return dict(
            start=self.start,
            end=self.end,
            page=self.page,
        )


@dataclass
class Box:
    l: float
    t: float
    w: float
    h: float
    page: int

    def to_json(self) -> Dict:
        return dict(l=self.l, t=self.t, w=self.w, h=self.h, page=self.page)


@dataclass
class SpanGroup:
    spans: List[Span] = field(default_factory=list)

    def to_json(self) -> List[Dict]:
        return [span.to_json() for span in self.spans]
    
    def __getitem__(self, key):
        return self.spans[key]


@dataclass
class BoxGroup:
    boxes: List[Box] = field(default_factory=list)

    def to_json(self) -> List[Dict]:
        return [box.to_json() for box in self.boxes]

    def __getitem__(self, key):
        return self.boxes[key]


@dataclass
class DocumentAnnotation:
    """DocumentAnnotation is intended for storing model predictions for a document."""

    doc: Optional["Document"] = field(default=False, init=False)
    # Specify an attribute with default value in the parent class
    # Ref: https://stackoverflow.com/a/58525728

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    @abstractmethod
    def annotate(self, doc: "Document"):
        """Annotate the object itself on a specific document. 
        It will associate the annotations with the document symbols. 
        """

    def __getattr__(self, field: str):
        if field in self.doc.fields: 
            return self.doc.find(self, field)
        else:
            return self.__getattribute__(field)


@dataclass
class DocSpanGroup(DocumentAnnotation):
    span_group: SpanGroup
    text: Optional[str] = None
    type: Optional[str] = None
    box_group: Optional[BoxGroup] = None

    def annotate(self, doc: "Document", field_name: str):
        """Annotate the object itself on a specific document. 
        It will associate the annotations with the document symbols. 
        """
        self.doc = doc
        doc_field_indexer = doc._indexers[field_name]

        for span in SpanGroup:

            # constraint - all spans disjoint
            existing = doc_field_indexer[span.page_id][span.start:span.end] 
            if existing:
                raise ValueError(f'Existing {existing} when attempting index {span}')
            # add to index
            doc_field_indexer[span.page_id][span.start:span.end] = self

    def to_json(self) -> Dict:
        return dict(
            _type="DocSpanGroup",  # Used for differenting between DocSpan and DocBox when loading the json
            span_group=self.span_group.to_json(),
            text=self.text,
            type=self.type,
            box_group=self.box_group.to_json() if self.box_group else None,
        )

@dataclass
class DocumentSpanAnnotationIndexer:

    num_pages: int

    def __post_init__(self):
        self._index: List[IntervalTree] = [IntervalTree() for _ in range(self.num_pages)]

    def __getitem__(self, indices):
        page_id, annotation_slice = indices
        return [interval.data for interval in self._index[page_id][annotation_slice]]

    def index(self, anno: DocSpanGroup):
        selected_annotations = []
        for anno in anno.span_group:
            for selected_anno in self._index[anno.page_id][anno.start : anno.end]:
                if selected_anno not in selected_annotations: # Deduplicate
                    selected_annotations.append(selected_anno) 
        return selected_annotations

@dataclass
class DocBoxGroup(DocumentAnnotation):
    box_group: BoxGroup
    type: Optional[str] = None

    def to_json(self) -> Dict:
        return dict(
            _type="DocBoxGroup",  # Used for differenting between DocSpan and DocBox when loading the json
            box_group=self.box_group.to_json(),
            type=self.type
        )



# Monkey patch the PIL.Image methods to add base64 conversion

def tobase64(self):
    # Ref: https://stackoverflow.com/a/31826470
    buffered = BytesIO()
    self.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode("utf-8")


def frombase64(img_str):
    # Use the same naming style as the original Image methods

    buffered = BytesIO(base64.b64decode(img_str))
    img = Image.open(buffered)
    return img  


Image.Image.tobase64 = tobase64 # This is the method applied to individual Image classes 
Image.Image.to_json = tobase64 # Use the same API as the others 
Image.frombase64 = frombase64 # This is bind to the module, used for loading the images 


def load_document_element(field_name: str, field_annotation: List[DocumentAnnotation], document: Optional["Document"] = None):
    pass