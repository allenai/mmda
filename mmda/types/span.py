"""



"""

from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class Span:
    start: int
    end: int
    page: int

    def to_json(self) -> Dict:
        return dict(start=self.start, end=self.end, page=self.page)


@dataclass
class SpanGroup:
    spans: List[Span] = field(default_factory=list)

    def to_json(self) -> List[Dict]:
        return [span.to_json() for span in self.spans]

    def __getitem__(self, key: int):
        return self.spans[key]
