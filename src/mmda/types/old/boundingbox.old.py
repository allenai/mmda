"""

Dataclass for doing stuff on bounding boxes

@kyle

"""

from typing import List, Dict

import json

class BoundingBox:
    def __init__(self, l: float, t: float, w: float, h: float, page: int):
        """Assumes x=0.0 and y=0.0 is the top-left of the page, and
           x=1.0 and y =1.0 as the bottom-right of the page"""
        if l < 0.0 or l > 1.0:
            raise ValueError(f'l={l} is not within 0.0~1.0')
        if t < 0.0 or t > 1.0:
            raise ValueError(f't={t} is not within 0.0~1.0')
        if l + w < 0.0 or l + w > 1.0:
            raise ValueError(f'l+w={l+w} is not within 0.0~1.0')
        if t + h < 0.0 or t + h > 1.0:
            raise ValueError(f't+h={t+h} is not within 0.0~1.0')
        self.l = l
        self.t = t
        self.w = w
        self.h = h
        self.page = page

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float,
                  page_height: float, page_width: float) -> 'BoundingBox':
        """Assumes (x0,y0) is top-left of box and (x1,y1) is bottom-right of box
        where x=0.0 and y=0.0 is top-left of the page"""
        raise NotImplementedError

    @classmethod
    def from_null(cls):
        """Creates an empty bbox; mostly useful for quick tests"""
        bbox = cls.__new__(cls)
        bbox.l = None
        bbox.t = None
        bbox.w = None
        bbox.h = None
        bbox.page = None
        return bbox

    @classmethod
    def from_json(cls, bbox_json: Dict) -> 'BoundingBox':
        l, t, w, h, page = bbox_json
        bbox = BoundingBox(l=l, t=t, w=w, h=h, page=page)
        return bbox

    def to_json(self):
        return [self.l, self.t, self.w, self.h, self.page]

    def __repr__(self):
        return json.dumps(self.to_json())

    @classmethod
    def union_bboxes(cls, bboxes: List['BoundingBox']) -> 'BoundingBox':
        if len({bbox.page for bbox in bboxes}) != 1:
            raise ValueError(f'Bboxes not all on same page: {bboxes}')
        x1 = min([bbox.l for bbox in bboxes])
        y1 = min([bbox.t for bbox in bboxes])
        x2 = max([bbox.l + bbox.w for bbox in bboxes])
        y2 = max([bbox.t + bbox.h for bbox in bboxes])
        return BoundingBox(page=bboxes[0].page, l=x1, t=y1, w=x2 - x1, h=y2 - y1)
