from typing import List, Optional, Type

from pydantic import BaseModel, Extra
from pydantic.fields import ModelField

import mmda.types.annotation as mmda_ann

__all__ = ["BoxGroup", "SpanGroup"]


class Box(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page: int

    @classmethod
    def from_mmda(cls, box: mmda_ann.Box) -> "Box":
        return cls(
            left=box.l, top=box.t, width=box.w, height=box.h, page=box.page
        )

    def to_mmda(self) -> mmda_ann.Box:
        return mmda_ann.Box(
            l=self.left,
            t=self.top,
            w=self.width,
            h=self.height,
            page=self.page
        )


class Span(BaseModel):
    start: int
    end: int
    box: Optional[Box]

    @classmethod
    def from_mmda(cls, span: mmda_ann.Span) -> "Span":
        box = Box.from_mmda(span.box) if span.box is not None else None
        ret = cls(start=span.start, end=span.end, box=box)
        if span.box is not None:
            ret.box = Box.from_mmda(span.box)
        return ret

    def to_mmda(self) -> mmda_ann.Span:
        return mmda_ann.Span.from_json(self.dict())


class Attributes(BaseModel):
    """Class to represent attributes for a SpanGroup or BoxGroup.
    Attributes are generally stored in mmda as mmda_ann.Metadata objects.

    Unlike mmda.types.annotation.Metadata, this class ignores fields
    id, type, and text, which are expected to be stored in the SpanGroup
    / BoxGroup itself."""

    @classmethod
    def from_mmda(cls, metadata: mmda_ann.Metadata) -> "Attributes":
        return cls(**metadata.to_json())

    def to_mmda(self) -> mmda_ann.Metadata:
        return mmda_ann.Metadata.from_json(self.dict())


class Annotation(BaseModel, extra=Extra.ignore):
    attributes: Attributes

    @classmethod
    def get_metadata_cls(cls) -> Type[Attributes]:
        attrs_field: ModelField = cls.__fields__["attributes"]
        return attrs_field.type_


class BoxGroup(Annotation):
    boxes: List[Box]
    id: Optional[int]
    type: Optional[str]

    @classmethod
    def from_mmda(cls, box_group: mmda_ann.BoxGroup) -> "BoxGroup":
        # for the Pydantic model, we need to convert the metadata to a dict,
        # and remove `id` and `type` that are stored there in MMDA
        # boxes need to be nestedly converted
        boxes = [Box.from_mmda(box) for box in box_group.boxes]
        metadata = cls.get_metadata_cls().from_mmda(box_group.metadata)

        return cls(
            boxes=boxes,
            id=box_group.id,
            type=box_group.type,
            attributes=metadata
        )

    def to_mmda(self) -> mmda_ann.BoxGroup:
        box_group_dict = self.dict()
        # we map what's in attributes to a metadata object
        box_group_dict["metadata"] = box_group_dict.pop("attributes", {})
        return mmda_ann.BoxGroup.from_json(box_group_dict)


class SpanGroup(Annotation):
    spans: List[Span]
    box_group: Optional[BoxGroup]
    id: Optional[int]
    type: Optional[str]
    text: Optional[str]

    @classmethod
    def from_mmda(cls, span_group: mmda_ann.SpanGroup) -> "SpanGroup":
        box_group = (
            BoxGroup.from_mmda(span_group.box_group)
            if span_group.box_group is not None
            else None
        )
        spans = [Span.from_mmda(sp) for sp in span_group.spans]
        metadata = cls.get_metadata_cls().from_mmda(span_group.metadata)

        ret = cls(
            spans=spans,
            box_group=box_group,
            id=span_group.id,
            type=span_group.type,
            text=span_group.text,
            attributes=metadata,
        )
        if span_group.box_group is not None:
            ret.box_group = BoxGroup.from_mmda(span_group.box_group)
        return ret

    def to_mmda(self) -> mmda_ann.SpanGroup:
        span_group_dict = self.dict()
        # we map what's in attributes to a metadata object
        span_group_dict["metadata"] = span_group_dict.pop("attributes", {})
        return mmda_ann.SpanGroup.from_json(span_group_dict)
