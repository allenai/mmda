"""

Dataclass for representing annotations on documents

@kylel

"""

from typing import List, Optional

import json


from mmda.types.tokens import Token, Span
from mmda.types.boundingbox import BoundingBox


class Annotation:
    def to_json(self):
        raise NotImplementedError

    def __repr__(self):
        return json.dumps(self.to_json())



class SpanAnnotation(Annotation):
    def __init__(self, span: Span, label: str):
        self.span = span
        self.label = label

    def to_json(self):
        return {'span': self.span.to_json(), 'label': self.label}

    def __contains__(self, i: int) -> bool:
        return i in self.span


class BoundingBoxAnnotation(Annotation):
    def __init__(self, bbox: BoundingBox, label: str):
        self.bbox = bbox
        self.label = label

    def to_json(self):
        return {'bbox': self.bbox.to_json(), 'label': self.label}



if __name__ == '__main__':

    # In this example, we construct a sequence tagger training dataset using these classes.

    tokens = ['I', 'live', 'in', 'New', 'York', '.', 'I', 'read', 'the', 'New', 'York', 'Times', '.']
    tokens = [Token.from_text(token) for token in tokens]
    spans = [Span(3, 5), Span(9, 12)]
    tags = [SpanAnnotation(span=span, label='entity') for span in spans]

    def get_label(i: int, tags: List[SpanAnnotation]) -> Optional[str]:
        for tag in tags:
            if i in tag:
                return tag.label
        return None

    training_data = []
    for i, token in enumerate(tokens):
        tag = get_label(i=i, tags=tags)
        if tag:
            training_data.append((token, 1))
        else:
            training_data.append((token, 0))
