"""



"""

from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field

from mmda.types.box import Box


@dataclass
class Span:
    start: int
    end: int
    box: Optional[Box] = None

    def to_json(self) -> Dict:
        if self.box:
            return dict(start=self.start, end=self.end, box=self.box.to_json())
        else:
            return dict(start=self.start, end=self.end)

    @classmethod
    def from_json(cls, span_dict) -> "Span":
        box_dict = span_dict.get('box')
        if box_dict:
            box = Box.from_json(box_coords=span_dict['box'])
        else:
            box = None
        return Span(start=span_dict['start'], end=span_dict['end'], box=box)
