from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, validator

import mmda.types.annotation as mmda_ann


__all__ = ["Span", "BoxGroup", "SpanGroup"]


def _validate_box(box: List[float]) -> None:
    assert len(box) == 5, "Box must have format [x, y, w, h, page]"
    assert int(box[4]) == box[4], "Page must be an integer value"


class Span(BaseModel):
    start: int
    end: int
    box: Optional[List[float]]

    @classmethod
    def from_mmda(cls, span: mmda_ann.Span) -> Span:
        ret = cls(start=span.start, end=span.end)
        if span.box is not None:
            ret.box = span.box.to_json()
        return ret

    def to_mmda(self) -> mmda_ann.Span:
        ret = mmda_ann.Span(start=self.start, end=self.end)
        if self.box is not None:
            ret.box = mmda_ann.Box.from_json(self.box)
        return ret

    @validator('box')
    def validate_box(cls, maybe_box):
        if maybe_box is not None:
            _validate_box(maybe_box)
        return maybe_box


class BoxGroup(BaseModel):
    boxes: List[List[float]]
    id: Optional[int]
    type: Optional[str]

    @classmethod
    def from_mmda(cls, box_group: mmda_ann.BoxGroup) -> BoxGroup:
        boxes = [box.to_json() for box in box_group.boxes]
        return cls(boxes=boxes, id=box_group.id, type=box_group.id)

    def to_mmda(self) -> mmda_ann.BoxGroup:
        boxes = [mmda_ann.Box.from_json(box) for box in self.boxes]
        return mmda_ann.BoxGroup(boxes=boxes, id=self.id, type=self.type)

    @validator('boxes')
    def validate_boxes(cls, boxes):
        for box in boxes:
            _validate_box(box)
        return boxes


class SpanGroup(BaseModel):
    spans: List[Span]
    box_group: Optional[BoxGroup]
    id: Optional[int]
    type: Optional[str]
    text: Optional[str]

    @classmethod
    def from_mmda(cls, span_group: mmda_ann.SpanGroup) -> SpanGroup:
        ret = cls(
            spans=[Span.from_mmda(sp) for sp in span_group.spans],
            id=span_group.id,
            type=span_group.type,
            text=span_group.text
        )
        if span_group.box_group is not None:
            ret.box_group = BoxGroup.from_mmda(span_group.box_group)
        return ret

    def to_mmda(self) -> mmda_ann.SpanGroup:
        ret = mmda_ann.SpanGroup(
            spans=[sp.to_mmda() for sp in self.spans],
            id=self.id,
            type=self.type,
            text=self.text
        )
        if self.box_group is not None:
            ret.box_group = self.box_group.to_mmda()
        return ret
