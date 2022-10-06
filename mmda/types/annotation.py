"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new
Iterable of Group-type objects within the Document

"""
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from mmda.types.box import Box
from mmda.types.metadata import Metadata, store_field_in_metadata
from mmda.types.span import Span

if TYPE_CHECKING:
    from mmda.types.document import Document


__all__ = ["Annotation", "BoxGroup", "SpanGroup", "Relation"]



def warn_deepcopy_of_annotation(obj: "Annotation") -> None:
    """Warns when a deepcopy is performed on an Annotation."""

    cls_name = type(obj).__name__
    msg = (
        f"Creating a deep copy of a {cls_name} is computationally"
        "expensive; consider using references instead."
    )
    warnings.warn(msg, UserWarning, stacklevel=2)



class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    def __init__(
            self,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            metadata: Optional[Metadata] = None
    ):
        self.id = id
        self.doc = doc
        self.metadata = metadata if metadata else Metadata()

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_dict: Dict) -> "Annotation":
        pass

    def attach_doc(self, doc: "Document") -> None:
        if not self.doc:
            self.doc = doc
        else:
            raise AttributeError("This annotation already has an attached document")

    # TODO[kylel] - comment explaining
    def __getattr__(self, field: str) -> List["Annotation"]:
        if self.doc is None:
            raise ValueError("This annotation is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        return self.__getattribute__(field)


# NOTE[LucaS]: by using the store_field_in_metadata decorator, we are
# able to store id and type in the metadata of BoxGroup, while keeping it
# accessible via SpanGroup.id and SpanGroup.type respectively. This is
# useful because it keeps backward compatibility with the old API, while
# migrating id and type to metadata.
@store_field_in_metadata("type")
class BoxGroup(Annotation):
    def __init__(
            self,
            boxes: List[Box],
            type: Optional[str] = None,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            metadata: Optional[Metadata] = None,
    ):
        self.boxes = boxes
        self.type = type
        super().__init__(id=id, doc=doc, metadata=metadata)

    def to_json(self) -> Dict:
        box_group_dict = dict(
            boxes=[box.to_json() for box in self.boxes],
            id=self.id,
            metadata=self.metadata.to_json()
        )
        return {
            key: value for key, value in box_group_dict.items() if value
        }  # only serialize non-null values

    @classmethod
    def from_json(cls, box_group_dict: Dict) -> "BoxGroup":

        if "metadata" in box_group_dict:
            metadata_dict = box_group_dict["metadata"]
        else:
            # this fallback is necessary to ensure compatibility with box
            # groups that were create before the metadata migration and
            # therefore have "id", "type" in the root of the json dict instead.
            metadata_dict = {
                "type": box_group_dict.get("type", None),
                "text": box_group_dict.get("text", None)
            }

        return cls(
            boxes=[
                Box.from_json(box_coords=box_dict)
                # box_group_dict["boxes"] might not be present since we
                # minimally serialize when running to_json()
                for box_dict in box_group_dict.get("boxes", [])
            ],
            id=box_group_dict.get("id", None),
            metadata=Metadata.from_json(metadata_dict),
        )

    def __getitem__(self, key: int):
        return self.boxes[key]

    def __deepcopy__(self, memo):
        warn_deepcopy_of_annotation(self)

        box_group = BoxGroup(
            boxes=deepcopy(self.boxes, memo),
            id=self.id,
            metadata=deepcopy(self.metadata, memo)
        )

        # Don't copy an attached document
        box_group.doc = self.doc

        return box_group


def _text_span_group_getter(span_group: "SpanGroup") -> str:
    """Getter used to obtain a textual representation of a SpanGroup.

    When SpanGroup.text is not set, this function uses the SpanGroup's
    symbols to generate approximate a text. However, if text is set,
    this function returns it instead.
    """
    maybe_text = span_group.metadata.get("text", None)
    return maybe_text if maybe_text else " ".join(span_group.symbols)


# NOTE[@soldni]: by using the store_field_in_metadata decorator, we are
# able to store id and type in the metadata of BoxGroup, while keeping it
# accessible via SpanGroup.id and SpanGroup.type respectively. This is
# useful because it keeps backward compatibility with the old API, while
# migrating id and type to metadata.
#
# Futhermore, we also store the text of the SpanGroup in the metadata,
# and use a custom getter to obtain the text from symbols if the text
# is not explicitly set.
@store_field_in_metadata("type")
@store_field_in_metadata("text", getter_fn=_text_span_group_getter)
class SpanGroup(Annotation):

    def __init__(
            self,
            spans: List[Span],
            type: Optional[str] = None,
            text: Optional[str] = None,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            metadata: Optional[Metadata] = None,
    ):
        self.spans = spans
        self.type = type
        self.text = text
        super().__init__(id=id, doc=doc, metadata=metadata)

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
            id=self.id,
            metadata=self.metadata.to_json(),
            box_group=self.box_group.to_json() if self.box_group else None
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

        if "metadata" in span_group_dict:
            metadata_dict = span_group_dict["metadata"]
        else:
            # this fallback is necessary to ensure compatibility with span
            # groups that were create before the metadata migration and
            # therefore have "id", "type" in the root of the json dict instead.
            metadata_dict = {
                "type": span_group_dict.get("type", None),
                "text": span_group_dict.get("text", None)
            }

        return cls(
            spans=[
                Span.from_json(span_dict=span_dict)
                for span_dict in span_group_dict["spans"]
            ],
            id=span_group_dict.get("id", None),
            metadata=Metadata.from_json(metadata_dict),
            box_group=box_group,
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
        warn_deepcopy_of_annotation(self)

        span_group = SpanGroup(
            spans=deepcopy(self.spans, memo),
            id=self.id,
            metadata=deepcopy(self.metadata, memo),
            box_group=deepcopy(self.box_group, memo)
        )

        # Don't copy an attached document
        span_group.doc = self.doc

        return span_group



class Relation(Annotation):
    pass