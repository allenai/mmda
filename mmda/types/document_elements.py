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
    def to_json() -> Dict:
        pass


@dataclass
class DocumentPageSymbols(DocumentElement):
    """Storing the symbols of a page."""

    symbols: str

    # TODO: Add support for symbol bounding boxes and style
    def __get_item__(self, key):
        return self.symbols[key]

    def to_json(self):
        return self.symbols


@dataclass
class DocumentSymbols(DocumentElement):
    """Storing the symbols of a document."""

    page_symbols: List[DocumentPageSymbols] = field(default_factory=list)

    def __get_item__(self, *indices):
        page_id, symbol_slices = indices
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

    def to_json(self) -> Dict:
        return [span.to_json() for span in self.spans]
    
    def __get_item__(self, key):
        return self.spans[key]


@dataclass
class BoxGroup:
    boxes: List[Box] = field(default_factory=list)

    def to_json(self) -> Dict:
        return [box.to_json() for box in self.boxes]

    def __get_item__(self, key):
        return self.boxes[key]

@dataclass
class DocumentAnnotation:
    """DocumentAnnotation is intended for storing model predictions for a document."""

    doc: "Document" = field(default=False, init=False) 
    # Specify an attribute with default value in the parent class
    # Ref: https://stackoverflow.com/a/58525728

    @abstractmethod
    def to_json() -> Dict:
        pass

    def __getattr__(self, field):
        if field in self.fields: 
            return self.doc.find(self, field)
        else:
            return self.__getattribute__(field)

@dataclass
class DocSpan(DocumentAnnotation):
    spans: List[Span] = field(default_factory=list)
    text: Optional[str] = None
    type: Optional[str] = None
    boxes: Optional["BoxGroup"] = None

    def to_json(self) -> Dict:
        return dict(
            _type="DocSpan",  # Used for differenting between DocSpan and DocBox when loading the json
            spans=self.spans.to_json(),
            page=self.page,
            text=self.text,
            type=self.type,
            boxes=self.boxes.to_json() if self.boxes else None,
        )


@dataclass
class DocBox(DocumentAnnotation):
    boxes: List[Box] = field(default_factory=list)
    text: Optional[str] = None
    type: Optional[str] = None
    spans: Optional["SpanGroup"] = None

    def to_json(self) -> Dict:
        return dict(
            _type="DocBox",  # Used for differenting between DocSpan and DocBox when loading the json
            boxes=self.boxes.to_json(),
            page=self.page,
            text=self.text,
            type=self.type,
            spans=self.spans.to_json() if self.spans else None,
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