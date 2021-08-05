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
    page: int
    box: Optional[Box] = None

    def __post_init__(self):
        assert self.box.page == self.page, f"Box.page {self.box.page} != Page {self.page}"

    def to_json(self) -> Dict:
        if self.box:
            return dict(start=self.start, end=self.end, page=self.page)
        else:
            return dict(start=self.start, end=self.end, page=self.page, box=self.box)

