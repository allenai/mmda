"""
Pydantic equivalents of classes from mmda.types
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator

from mmda.types.document import Document as MmdaDoc, SpanGroup as MmdaSpanGroup


class Span(BaseModel):
    start: int
    end: int
    box: Optional[List[float]] = Field(max_items=5, min_items=5)


class BoxGroup(BaseModel):
    boxes: List[List[float]]
    type: Optional[str]

    @validator("boxes", each_item=True)
    def check_box(cls, v):
        assert len(v) == 5, "Box must have format [x, y, w, h, page]"
        assert int(v[4]) == v[4], "Page must be an integer value"
        return v


class SpanGroup(BaseModel):
    spans: List[Span]
    type: Optional[str]
    box_group: Optional[BoxGroup]

    @classmethod
    def from_mmda(cls, grp: MmdaSpanGroup):
        return SpanGroup.parse_obj(grp.to_json())


class Document(BaseModel):
    symbols: str
    tokens: List[SpanGroup]
    pages: List[SpanGroup]
    rows: List[SpanGroup]
    images: Optional[List[str]]
    blocks: Optional[List[SpanGroup]]

    def to_mmda(self) -> MmdaDoc:
        return MmdaDoc.from_json(self.dict())
