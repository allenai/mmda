"""

Dataclass for doing stuff on token streams of a document

@kylel

"""

from typing import List, Optional, Dict, Union

from mmda.types.boundingbox import BoundingBox

import json


class Span:
    def __init__(self, start: int, end: int, text: Optional[str] = None, bbox: Optional[BoundingBox] = None):
        self.start = start
        self.end = end
        self.text = text
        self.bbox = bbox

    @classmethod
    def from_json(cls, span_json: Dict) -> 'Span':
        bbox = BoundingBox.from_json(bbox_json=span_json['bbox']) if 'bbox' in span_json else None
        span = Span(start=span_json['start'], end=span_json['end'], text=span_json.get('text'), bbox=bbox)
        return span

    def to_json(self):
        return {'start': self.start,
                'end': self.end,
                'text': self.text if self.text else None,
                'bbox': self.bbox.to_json() if self.bbox else None}

    def __repr__(self):
        return json.dumps(self.to_json())

    def __contains__(self, val: Union[int, BoundingBox]) -> bool:
        """Checks whether an index value `i` is within the span"""
        if isinstance(val, int):
            return self.start <= val < self.end
        elif isinstance(val, str):
            return val in self.text
        elif isinstance(val, BoundingBox):
            raise NotImplementedError
        else:
            raise ValueError(f'{val} of type {type(val)} not supported for __contains__')

