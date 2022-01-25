from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

import mmda.types.annotation as mmda_ann


__all__ = ["BoxGroup", "SpanGroup"]


class Box(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page: int

    @classmethod
    def from_mmda(cls, box: mmda_ann.Box) -> Box:
        return cls(
            left=box.l, top=box.t, width=box.w, height=box.h, page=box.page
        )

    def to_mmda(self) -> mmda_ann.Box:
        return mmda_ann.Box(
            l=self.left, t=self.top, w=self.width, h=self.height, page=self.page
        )


class Span(BaseModel):
    start: int
    end: int
    box: Optional[Box]

    @classmethod
    def from_mmda(cls, span: mmda_ann.Span) -> Span:
        ret = cls(start=span.start, end=span.end)
        if span.box is not None:
            ret.box = Box.from_mmda(span.box)
        return ret

    def to_mmda(self) -> mmda_ann.Span:
        ret = mmda_ann.Span(start=self.start, end=self.end)
        if self.box is not None:
            ret.box = self.box.to_mmda()
        return ret


class BoxGroup(BaseModel):
    boxes: List[Box]
    id: Optional[int]
    type: Optional[str]

    @classmethod
    def from_mmda(cls, box_group: mmda_ann.BoxGroup) -> BoxGroup:
        boxes = [Box.from_mmda(box) for box in box_group.boxes]
        return cls(boxes=boxes, id=box_group.id, type=box_group.id)

    def to_mmda(self) -> mmda_ann.BoxGroup:
        boxes = [box.to_mmda() for box in self.boxes]
        return mmda_ann.BoxGroup(boxes=boxes, id=self.id, type=self.type)


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
