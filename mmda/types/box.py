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

    def to_json(self) -> Dict:
        return dict(l=self.l, t=self.t, w=self.w, h=self.h, page=self.page)


@dataclass
class BoxGroup:
    boxes: List[Box] = field(default_factory=list)

    def to_json(self) -> List[Dict]:
        return [box.to_json() for box in self.boxes]

    def __getitem__(self, key: int):
        return self.boxes[key]