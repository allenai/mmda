"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new Iterable of Group-type objects within the Document

"""

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from mmda.types.box import Box
from mmda.types.names import Symbols
from mmda.types.span import Span


@dataclass
class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    uuid: str = field(default_factory=uuid4)
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

    @property
    def key_prefix(self) -> str:
        return f"{self.__class__.__name__}|{self.uuid}|"

    def attach_doc(self, doc: "Document") -> None:
        if not self.doc:
            self.doc = doc
        else:
            raise AttributeError(f"This annotation already has an attached document")

    # TODO[kylel] - comment explaining
    def __getattr__(self, field: str) -> List["Annotation"]:
        if self.key_prefix + field in self.doc.fields:
            return self.doc.find_overlapping(self, self.key_prefix + field)

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        return self.__getattribute__(field)


@dataclass
class BoxGroup(Annotation):
    boxes: List[Box] = field(default_factory=list)
    id: Optional[int] = None
    type: Optional[str] = None

    def to_json(self) -> Dict:
        box_group_dict = dict(
            boxes=[box.to_json() for box in self.boxes],
            id=self.id,
            type=self.type,
            uuid=self.uuid,
        )
        return {
            key: value for key, value in box_group_dict.items() if value
        }  # only serialize non-null values

    @classmethod
    def from_json(cls, box_group_dict: Dict) -> "BoxGroup":
        return BoxGroup(
            boxes=[
                Box.from_json(box_coords=box_dict)
                for box_dict in box_group_dict["boxes"]
            ],
            id=box_group_dict.get("id"),
            type=box_group_dict.get("type"),
            uuid=box_group_dict.get("uuid", uuid4()),
        )

    def __getitem__(self, key: int):
        return self.boxes[key]

    def __deepcopy__(self, memo):
        box_group = BoxGroup(
            id=self.id, boxes=deepcopy(self.boxes, memo), type=self.type, uuid=self.uuid
        )

        # Don't copy an attached document
        box_group.doc = self.doc

        return box_group


@dataclass
class SpanGroup(Annotation):
    spans: List[Span] = field(default_factory=list)
    id: Optional[int] = None
    text: Optional[str] = None
    type: Optional[str] = None
    box_group: Optional[BoxGroup] = None  # TODO[kylel] - implement default behavior

    @property
    def symbols(self) -> List[str]:
        return [self.doc.symbols[span.start : span.end] for span in self.spans]

    def annotate(
        self, is_overwrite: bool = False, **kwargs: Iterable["Annotation"]
    ) -> None:
        if self.doc is None:
            raise ValueError("SpanGroup has no attached document!")

        key_remaps = {self.key_prefix + k: v for k, v in kwargs.items()}

        self.doc.annotate(is_overwrite=is_overwrite, **key_remaps)

    def to_json(self) -> Dict:
        span_group_dict = dict(
            spans=[span.to_json() for span in self.spans],
            id=self.id,
            text=self.text,
            type=self.type,
            box_group=self.box_group.to_json() if self.box_group else None,
            uuid=self.uuid,
        )
        return {
            key: value for key, value in span_group_dict.items() if value is not None
        }  # only serialize non-null values

    @classmethod
    def from_json(cls, span_group_dict: Dict) -> "SpanGroup":
        box_group_dict = span_group_dict.get("box_group")
        if box_group_dict:
            box_group = BoxGroup.from_json(box_group_dict=box_group_dict)
        else:
            box_group = None
        return SpanGroup(
            spans=[
                Span.from_json(span_dict=span_dict)
                for span_dict in span_group_dict["spans"]
            ],
            id=span_group_dict.get("id"),
            text=span_group_dict.get("text"),
            type=span_group_dict.get("type"),
            box_group=box_group,
            uuid=span_group_dict.get("uuid", uuid4()),
        )

    def __getitem__(self, key: int):
        return self.spans[key]

    @property
    def start(self) -> int:
        return (
            min([span.start for span in self.spans])
            if len(self.spans) > 0
            else float("-inf")
        )

    @property
    def end(self) -> int:
        return (
            max([span.end for span in self.spans])
            if len(self.spans) > 0
            else float("inf")
        )

    def __lt__(self, other: "SpanGroup"):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start

    def __deepcopy__(self, memo):
        span_group = SpanGroup(
            spans=deepcopy(self.spans, memo),
            id=self.id,
            text=self.text,
            type=self.type,
            box_group=deepcopy(self.box_group, memo),
            uuid=self.uuid,
        )

        # Don't copy an attached document
        span_group.doc = self.doc

        return span_group
