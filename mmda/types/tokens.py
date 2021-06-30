"""

Dataclass for doing stuff on token streams of a document

@kyle

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


class Tag:
    def __init__(self, span: Span, label: str):
        self.span = span
        self.label = label

    def to_json(self):
        return {'span': self.span.to_json(), 'label': self.label}

    def __repr__(self):
        return json.dumps(self.to_json())

    def __contains__(self, i: int) -> bool:
        return i in self.span


if __name__ == '__main__':
    tokens = ['I', 'live', 'in', 'New', 'York', '.', 'I', 'read', 'the', 'New', 'York', 'Times', '.']
    tokens = [Token.from_text(token) for token in tokens]
    spans = [Span(3, 5), Span(9, 12)]
    tags = [Tag(span=span, label='entity') for span in spans]


    def get_tag(i: int, tags: List[Tag]) -> Optional[Tag]:
        for tag in tags:
            if i in tag:
                return tag
        return None


    training_data = []
    for i, token in enumerate(tokens):
        tag = get_tag(i=i, tags=tags)
        if tag:
            training_data.append((token, 1))
        else:
            training_data.append((token, 0))
