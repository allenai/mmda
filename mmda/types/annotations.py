"""

Dataclass for representing annotations on documents

@kylel

"""

from typing import List, Optional

import json


from mmda.types.span import Span
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

    text = 'I live in New York. I read the New York Times.'
    tokens = [(0, 1), (2, 6), (7, 9), (10, 13), (14, 18), (18, 19),
              (20, 21), (22, 26), (27, 30), (31, 34), (35, 39), (40, 45), (45, 46)]
    tokens = [Span(start=start, end=end) for start, end in tokens]
    for token in tokens:
        print(text[token.start:token.end])

    tags = [Span(start=10, end=19, attr=['entity']), Span(start=31, end=46, attr=['entity'])]
    for tag in tags:
        print(f'{text[tag.start:tag.end]}\t{tag.tags}')

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
