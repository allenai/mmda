"""

Dataclass for doing stuff on token streams of a document

@kylel

"""

from typing import List, Optional, Dict, Union

from mmda.types.boundingbox import BoundingBox

import json


class Span:
    def __init__(self, start: int, end: int, id: Optional[int] = None, type: Optional[str] = None,
                 text: Optional[str] = None, bbox: Optional[BoundingBox] = None):
        self.start = start
        self.end = end
        self.type = type
        self.id = id
        self.text = text
        self.bbox = bbox

    @classmethod
    def from_json(cls, span_json: Dict):
        bbox = BoundingBox.from_json(bbox_json=span_json['bbox']) if 'bbox' in span_json else None
        span = cls(start=span_json['start'], end=span_json['end'], id=span_json.get('id'),
                   type=span_json.get('type'), text=span_json.get('text'), bbox=bbox)
        return span

    def to_json(self, exclude: List[str] = []) -> Dict:
        full_json = {'start': self.start,
                     'end': self.end,
                     'type': self.type,
                     'id': self.id,
                     'text': self.text,
                     'bbox': self.bbox.to_json() if self.bbox else None}
        # the `is not None` is to save serialization space for empty fields
        return {k: v for k, v in full_json.items() if k not in exclude and v is not None}

    def __repr__(self):
        return json.dumps({k: v for k, v in self.to_json().items() if v is not None})

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

    def __lt__(self, other: 'Span'):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start
