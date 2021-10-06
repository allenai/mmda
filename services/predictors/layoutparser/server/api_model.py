from typing import List, Optional

from pydantic import BaseModel, validator, Field, ValidationError
from mmda.types.annotation import BoxGroup as MmdaBoxGroup


class Instance(BaseModel):
    """Input is a list of page images, base64-encoded"""

    images: List[str] = Field(description="List of base64-encoded page images")


class BoxGroup(BaseModel):
    boxes: List[List[float]]
    type: Optional[str]

    def to_mmda(self):
        return MmdaBoxGroup.from_json(self.dict())

    @validator("boxes", each_item=True)
    def check_box(cls, v):
        assert len(v) == 5, "Box must have format [x, y, w, h, page]"
        assert int(v[4]) == v[4], "Page must be an integer value"
        v[4] = int(v[4])
        return v

    @classmethod
    def from_mmda(cls, grp):
        return BoxGroup.parse_obj(grp.to_json())
