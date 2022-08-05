"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new
Iterable of Group-type objects within the Document

"""
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

from mmda.types.box import Box
from mmda.types.decorators import store_field_in_metadata
from mmda.types.span import Span

if TYPE_CHECKING:
    from mmda.types.document import Document


def default_factory():
    return str(uuid4())


@dataclass
class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    # TODO[kylel] - remove UUID from this class, as you explained to me (luca)
    # it is about 10% of the wall time in processing a document
    uuid: str = field(default_factory=default_factory)
    doc: Optional["Document"] = field(default=None, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    # TODO[shannon] make this as an abstract method after implementing
    # get_symbols for BoxGroup
    def get_symbols(self) -> str:  # type: ignore
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
            raise AttributeError(
                "This annotation already has an attached document"
            )

    # TODO[kylel] - comment explaining
    def __getattr__(self, field: str) -> List["Annotation"]:
        if self.doc is None:
            raise ValueError("This annotation is not attached to a document")

        if self.key_prefix + field in self.doc.fields:
            return self.doc.find_overlapping(self, self.key_prefix + field)

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        return self.__getattribute__(field)


# NOTE[LucaS]: by using the store_field_in_metadata decorator, we are
# able to store id and type in the metadata of BoxGroup, while keeping it
# accessible via SpanGroup.id and SpanGroup.type respectively. This is
# useful because it keeps backward compatibility with the old API, while
# migrating id and type to metadata.
@store_field_in_metadata("type")
@store_field_in_metadata("id")
@dataclass
class BoxGroup(Annotation):
    boxes: List[Box] = field(default_factory=list)
    id: Optional[int] = None
    type: Optional[str] = None

    def to_json(self) -> Dict:
        box_group_dict = dict(
            boxes=[box.to_json() for box in self.boxes],
            metadata=self.metadata,
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
            metadata=box_group_dict.get(
                "metadata",
                # this is necessary to ensure compatibility with
                # box groups that were create before the metadata
                # migration and therefore have "id", "type" in the
                # main representation instead.
                {
                    key: box_group_dict[key]
                    for key in ("id", "type")
                    if key in box_group_dict
                }
            ),
            uuid=box_group_dict.get("uuid", str(uuid4())),
        )

    def __getitem__(self, key: int):
        return self.boxes[key]

    def __deepcopy__(self, memo):
        box_group = BoxGroup(
            boxes=deepcopy(self.boxes, memo),
            metadata=deepcopy(self.metadata, memo),
            uuid=self.uuid,
        )

        # Don't copy an attached document
        box_group.doc = self.doc

        return box_group


def _text_span_group_getter(span_group: 'SpanGroup') -> str:
    """Getter used to obtain a textual representation of a SpanGroup.

    When SpanGroup.text is not set, this function uses the SpanGroup's
    symbols to generate approximate a text. However, if text is set,
    this function returns it instead.
    """

    if text := span_group.metadata.get("text", None):
        return text
    else:
        return " ".join(span_group.symbols)


# NOTE[LucaS]: by using the store_field_in_metadata decorator, we are
# able to store id and type in the metadata of BoxGroup, while keeping it
# accessible via SpanGroup.id and SpanGroup.type respectively. This is
# useful because it keeps backward compatibility with the old API, while
# migrating id and type to metadata.
#
# Futhermore, we also store the text of the SpanGroup in the metadata,
# and use a custom getter to obtain the text from symbols if the text
# is not explicitly set.
@store_field_in_metadata("type")
@store_field_in_metadata("id")
@store_field_in_metadata("text", getter_fn=_text_span_group_getter)
@dataclass
class SpanGroup(Annotation):
    spans: List[Span] = field(default_factory=list)

    # TODO[kylel] - implement default behavior for box_group
    box_group: Optional[BoxGroup] = None

    id: Optional[int] = None
    type: Optional[str] = None
    text: Optional[str] = None

    @property
    def symbols(self) -> List[str]:
        if self.doc is not None:
            return [
                self.doc.symbols[span.start: span.end] for span in self.spans
            ]
        else:
            return []

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
            metadata=self.metadata,
            box_group=self.box_group.to_json() if self.box_group else None,
            uuid=self.uuid,
        )
        return {
            key: value
            for key, value in span_group_dict.items()
            if value is not None
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
            metadata=span_group_dict.get(
                "metadata",
                # this is necessary to ensure compatibility with
                # span groups that were create before the metadata
                # migration and therefore have "id", "type", and "text"
                # in the main representation instead.
                {
                    key: span_group_dict[key]
                    for key in ("id", "type", "text")
                    if key in span_group_dict
                }
            ),
            box_group=box_group,
            uuid=span_group_dict.get("uuid", str(uuid4())),
        )

    def __getitem__(self, key: int):
        return self.spans[key]

    @property
    def start(self) -> Union[int, float]:
        return (
            min([span.start for span in self.spans])
            if len(self.spans) > 0
            else float("-inf")
        )

    @property
    def end(self) -> Union[int, float]:
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
            metadata=deepcopy(self.metadata, memo),
            box_group=deepcopy(self.box_group, memo),
            uuid=self.uuid,
        )

        # Don't copy an attached document
        span_group.doc = self.doc

        return span_group
