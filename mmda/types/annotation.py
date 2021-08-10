"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new Iterable of Group-type objects within the Document

"""

from mmda.types.names import Symbols
from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field


from mmda.types.span import Span
from mmda.types.box import Box

from intervaltree import IntervalTree


@dataclass
class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    doc: Optional["Document"] = field(default=False, init=False)
    # Specify an attribute with default value in the parent class
    # Ref: https://stackoverflow.com/a/58525728

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    # TODO[shannon] make this as an abstract method after implementing
    # get_symbols for BoxGroup
    def get_symbols(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_dict: Dict) -> "Annotation":
        pass

    def attach_doc(self, doc: "Document") -> None:
        if not self.doc:
            self.doc = doc
        else:
            raise AttributeError(f'This annotation already has an attached document')

    # TODO[kylel] - comment explaining
    def __getattr__(self, field: str) -> List["Annotation"]:
        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)
        elif field == Symbols:
            return self.get_symbols()
        else:
            return self.__getattribute__(field)     # TODO[kylel] - alternatively, have it fail


@dataclass
class BoxGroup(Annotation):
    boxes: List[Box] = field(default_factory=list)
    id: Optional[int] = None
    type: Optional[str] = None

    def to_json(self) -> Dict:
        box_group_dict = dict(
            boxes=[box.to_json() for box in self.boxes],
            id=self.id,
            type=self.type
        )
        return {key: value for key, value in box_group_dict.items() if value}  # only serialize non-null values

    @classmethod
    def from_json(cls, box_group_dict: Dict) -> "BoxGroup":
        return BoxGroup(boxes=[Box.from_json(box_coords=box_dict) for box_dict in box_group_dict['boxes']],
                        id=box_group_dict.get('id'),
                        type=box_group_dict.get('type'))

    def __getitem__(self, key: int):
        return self.boxes[key]


@dataclass
class SpanGroup(Annotation):
    spans: List[Span] = field(default_factory=list)
    id: Optional[int] = None
    text: Optional[str] = None
    type: Optional[str] = None
    box_group: Optional[BoxGroup] = None  # TODO[kylel] - implement default behavior

    def get_symbols(self) -> str:
        if self.text is not None:
            return self.text
        else:
            return ' '.join([self.doc.symbols[span.start:span.end] for span in self.spans])    

    def to_json(self) -> Dict:
        span_group_dict = dict(
            spans=[span.to_json() for span in self.spans],
            id=self.id,
            text=self.text,
            type=self.type,
            box_group=self.box_group.to_json() if self.box_group else None,
        )
        return {key: value for key, value in span_group_dict.items() if value is not None}  # only serialize non-null values

    @classmethod
    def from_json(cls, span_group_dict: Dict) -> "SpanGroup":
        box_group_dict = span_group_dict.get('box_group')
        if box_group_dict:
            box_group = BoxGroup.from_json(box_group_dict=box_group_dict)
        else:
            box_group = None
        return SpanGroup(spans=[Span.from_json(span_dict=span_dict) for span_dict in span_group_dict['spans']],
                         id=span_group_dict.get('id'),
                         text=span_group_dict.get('text'),
                         type=span_group_dict.get('type'),
                         box_group=box_group)

    def __getitem__(self, key: int):
        return self.spans[key]


# TODO[kylel] -- Implement
@dataclass
class Indexer:
    """Stores an index for a particular collection of Annotations.
    Indexes in this library focus on *INTERSECT* relations."""

    @abstractmethod
    def find(self, query: Annotation) -> List[Annotation]:
        """Returns all matching Annotations given a suitable query"""


@dataclass
class SpanGroupIndexer(Indexer):

    # careful; if write it as _index = IntervalTree(), all SpanGroupIndexers will share the same _index object
    _index: IntervalTree = field(default_factory=IntervalTree)

    # TODO[kylel] - maybe have more nullable args for different types of queryes (just start/end ints, just SpanGroup)
    def find(self, query: SpanGroup) -> List[SpanGroup]:
        if not isinstance(query, SpanGroup):
            raise ValueError(f'SpanGroupIndexer only works with `query` that is SpanGroup type')

        all_matched_spans = []
        for span in query.spans:
            for matched_span in self._index[span.start : span.end]:
                if matched_span.data not in all_matched_spans: # Deduplicate
                    all_matched_spans.append(matched_span.data)
        return all_matched_spans

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key, value):
        self._index[key] = value
