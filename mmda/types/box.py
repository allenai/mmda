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

