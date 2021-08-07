"""



"""


from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class Box:
    l: float
    t: float
    w: float
    h: float
    page: int

    # TODO[kylel]: make this more minimal
    def to_json(self) -> Dict:
        return dict(l=self.l, t=self.t, w=self.w, h=self.h, page=self.page)

    @classmethod
    def from_json(cls, box_dict) -> "Box":
        return Box(l=box_dict['l'], t=box_dict['t'], w=box_dict['w'], h=box_dict['h'], page=box_dict['page'])

    @classmethod
    def small_boxes_to_big_box(cls, boxes: List['Box']) -> 'Box':
        """Computes one big box that tightly encapsulates all smaller input boxes"""
        if len({box.page for box in boxes}) != 1:
            raise ValueError(f'Bboxes not all on same page: {boxes}')
        x1 = min([bbox.l for bbox in boxes])
        y1 = min([bbox.t for bbox in boxes])
        x2 = max([bbox.l + bbox.w for bbox in boxes])
        y2 = max([bbox.t + bbox.h for bbox in boxes])
        return Box(page=boxes[0].page, l=x1, t=y1, w=x2 - x1, h=y2 - y1)
