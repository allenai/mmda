"""

Dataclass for doing stuff on token streams of a document

@kyle

"""

from typing import List

from mmda.types.boundingbox import BoundingBox

import json


class Token:
    def __init__(self, text: str, bbox: BoundingBox):
        self.text = text
        self.bbox = bbox

    def to_json(self):
        return {'text': self.text, 'bbox': self.bbox.to_json()}

    @classmethod
    def from_text(cls, text: str) -> 'Token':
        return Token(text=text, bbox=BoundingBox.from_null())

    def __repr__(self):
        return json.dumps(self.to_json())



class TokenStream:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.tokens):
            result = self.tokens[self.i]
            self.i += 1
            return result
        else:
            raise StopIteration

    def to_json(self):
        return [token.to_json() for token in self.tokens]

    def __repr__(self):
        return json.dumps(self.to_json())