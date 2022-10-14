"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new
Iterable of Group-type objects within the Document

@kylel, @lucas

"""
import logging
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from mmda.types.box import Box
from mmda.types.metadata import Metadata
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


class AnnotationName:
    """Stores a name that uniquely identifies this Annotation within a Document"""

    def __init__(self, field: str, id: int):
        self.field = field
        self.id = id

    def __str__(self) -> str:
        return f"{self.field}-{self.id}"

    @classmethod
    def from_str(cls, s: str) -> 'AnnotationName':
        field, id = s.split('-')
        id = int(id)
        return AnnotationName(field=field, id=id)


class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    def __init__(
            self,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            field: Optional[str] = None,
            metadata: Optional[Metadata] = None
    ):
        self.id = id
        self.doc = doc
        self.field = field
        self.metadata = metadata if metadata else Metadata()

    @abstractmethod
    def to_json(self) -> Dict:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_dict: Dict) -> "Annotation":
        raise NotImplementedError

    @property
    def name(self) -> Optional[AnnotationName]:
        if self.field and self.id:
            return AnnotationName(field=self.field, id=self.id)
        else:
            return None

    def _attach_doc(self, doc: "Document", field: str) -> None:
        if not self.doc:
            self.doc = doc
            self.field = field
        else:
            raise AttributeError("This annotation already has an attached document")

    def _get_siblings(self) -> List['Annotation']:
        """This method gets all other objects sharing the same field as the current object.
        Only works after a Document has been attached, which is how objects learn their `field`."""
        if not self.doc:
            raise AttributeError("This annotation does not have an attached document")
        return self.doc.__getattribute__(self.field)

    def __getattr__(self, field: str) -> List["Annotation"]:
        """This method allows jumping from an object of one field to all overlapping
        objects of another field. For example `page.tokens` jumps from a particular page
        to all its intersecting tokens."""
        if not self.doc:
            raise AttributeError("This annotation does not have an attached document")

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        # TODO[kylel] - when does this ever get called? infinite loop?
        return self.__getattribute__(field)


class BoxGroup(Annotation):
    def __init__(
            self,
            boxes: List[Box],
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            field: Optional[str] = None,
            metadata: Optional[Metadata] = None,
    ):
        self.boxes = boxes
        super().__init__(id=id, doc=doc, field=field, metadata=metadata)

    def to_json(self) -> Dict:
        """Note: even if `doc` or `field` are attached, don't include in JSON to avoid bloat"""
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
            # therefore have "type" in the root of the json dict instead.
            metadata_dict = {
                "type": box_group_dict.get("type", None)
            }

        return cls(
            boxes=[
                Box.from_json(box_dict=box_dict)
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
            field=self.field,
            metadata=deepcopy(self.metadata, memo)
        )

        # Don't copy an attached document
        box_group.doc = self.doc

        return box_group

    @property
    def type(self) -> str:
        logging.warning(msg='`.type` to be deprecated in future versions. Use `.metadata.type`')
        return self.metadata.get("type", None)

    @type.setter
    def type(self, type: Union[str, None]) -> None:
        logging.warning(msg='`.type` to be deprecated in future versions. Use `.metadata.type`')
        self.metadata.type = type


class SpanGroup(Annotation):
    def __init__(
            self,
            spans: List[Span],
            box_group: Optional[BoxGroup] = None,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            field: Optional[str] = None,
            metadata: Optional[Metadata] = None,
    ):
        self.spans = spans
        self.box_group = box_group
        super().__init__(id=id, doc=doc, field=field, metadata=metadata)

    @property
    def symbols(self) -> List[str]:
        if self.doc is not None:
            return [self.doc.symbols[span.start: span.end] for span in self.spans]
        else:
            return []

    def to_json(self) -> Dict:
        """Note: even if `doc` or `field` are attached, don't include in JSON to avoid bloat"""
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
            # therefore have "type" in the root of the json dict instead.
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
            field=self.field,
            metadata=deepcopy(self.metadata, memo),
            box_group=deepcopy(self.box_group, memo)
        )

        # Don't copy an attached document
        span_group.doc = self.doc

        return span_group

    @property
    def type(self) -> str:
        logging.warning(msg='`.type` to be deprecated in future versions. Use `.metadata.type`')
        return self.metadata.get("type", None)

    @type.setter
    def type(self, type: Union[str, None]) -> None:
        logging.warning(msg='`.type` to be deprecated in future versions. Use `.metadata.type`')
        self.metadata.type = type

    @property
    def text(self) -> str:
        maybe_text = self.metadata.get("text", None)
        if maybe_text is None:
            return " ".join(self.symbols)
        return maybe_text

    @text.setter
    def text(self, text: Union[str, None]) -> None:
        self.metadata.text = text


class Relation(Annotation):
    def __init__(
            self,
            key: SpanGroup,
            value: SpanGroup,
            id: Optional[int] = None,
            doc: Optional['Document'] = None,
            field: Optional[str] = None,
            metadata: Optional[Metadata] = None
    ):
        if key.name is None:
            raise ValueError(f'Relation requires the key {key} to have a `.name`')
        if value.name is None:
            raise ValueError(f'Relation requires the value {value} to have a `.name`')
        self.key = key
        self.value = value
        super().__init__(id=id, doc=doc, field=field, metadata=metadata)

    def to_json(self) -> Dict:
        """Note: even if `doc` or `field` are attached, don't include in JSON to avoid bloat"""
        relation_dict = dict(
            key=str(self.key.name),
            value=str(self.value.name),
            id=self.id,
            metadata=self.metadata.to_json()
        )
        return {
            key: value
            for key, value in relation_dict.items()
            if value is not None
        }  # only serialize non-null values

    @classmethod
    def from_json(
            cls,
            relation_dict: Dict,
            doc: 'Document',
    ) -> "Relation":
        key_name = AnnotationName.from_str(s=relation_dict['key'])
        value_name = AnnotationName.from_str(s=relation_dict['value'])
        return cls(
            key=doc.locate_annotation(name=key_name),
            value=doc.locate_annotation(name=value_name),
            id=relation_dict.get("id", None),
            metadata=Metadata.from_json(relation_dict.get('metadata', {}))
        )
