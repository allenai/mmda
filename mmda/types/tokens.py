"""

Dataclass for doing stuff on token streams of a document

@kylel

"""

from typing import List, Optional

from mmda.types.boundingbox import BoundingBox

import json


class Token:
    def __init__(self, text: str, bbox: BoundingBox):
        self.text = text
        self.bbox = bbox

    def to_json(self):
        return {'text': self.text, 'bbox': self.bbox.to_json()}

    def __repr__(self):
        return json.dumps(self.to_json())

    @classmethod
    def from_text(cls, text: str) -> 'Token':
        return Token(text=text, bbox=BoundingBox.from_null())


class Span:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def to_json(self):
        return [self.start, self.end]

    def __repr__(self):
        return json.dumps(self.to_json())

    def __contains__(self, i: int) -> bool:
        """Checks whether an index value `i` is within the span"""
        return self.start <= i < self.end


class Row:
    raise NotImplementedError


class Block:
    raise NotImplementedError


