"""

Dataclass for representing a document and all its constituents

@kylel

"""

from typing import List

from mmda.types.annotations import Annotation, SpanAnnotation, BoundingBoxAnnotation
from mmda.types.tokens import Token


class Document:
    def annotate(self, annotations: List[Annotation]):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError


class TextDocument(Document):
    # TODO: organize things like visual block regions, rows, etc.
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens

    # TODO: what happens to the document data when annotate?
    def annotate(self, annotations: List[Annotation]):
        for annotation in annotations:
            if isinstance(annotation, SpanAnnotation):
                pass
            elif isinstance(annotation, BoundingBoxAnnotation):
                pass
            else:
                pass
        raise NotImplementedError


class ImgDocument(Document):
    raise NotImplementedError


class HybridDocument(Document):
    raise NotImplementedError