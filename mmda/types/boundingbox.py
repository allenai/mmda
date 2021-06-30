"""

Dataclass for doing stuff on bounding boxes

@kyle

"""

import json

class BoundingBox:
    def __init__(self):
        raise NotImplementedError

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, page: int,
                  page_height: float, page_width: float) -> 'BoundingBox':
        """Assumes x=0.0 and y=0.0 is the top-left of the page"""
        bbox = cls.__new__(cls)
        bbox.x = x / page_width
        bbox.y = y / page_height
        bbox.w = w / page_width
        bbox.h = h / page_height
        bbox.page = page
        return bbox

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float,
                  page_height: float, page_width: float) -> 'BoundingBox':
        """Assumes (x0,y0) is top-left of box and (x1,y1) is bottom-right of box
        where x=0.0 and y=0.0 is top-left of the page"""
        raise NotImplementedError

    @classmethod
    def from_null(cls):
        """Creates an empty bbox"""
        bbox = cls.__new__(cls)
        bbox.x = None
        bbox.y = None
        bbox.w = None
        bbox.h = None
        bbox.page = None
        return bbox

    def to_json(self):
        return [self.x, self.y, self.w, self.h]

    def __repr__(self):
        return json.dumps(self.to_json())