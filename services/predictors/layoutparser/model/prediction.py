from typing import List
from pydantic import BaseModel, Field


class Box(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page: int


class BoxGroup(BaseModel):
    boxes: List[Box]
    type: str


class Prediction(BaseModel):
    """Output is a set of bounding boxes with metadata"""

    groups: List[BoxGroup] = Field(description="PDF Text Regions")
