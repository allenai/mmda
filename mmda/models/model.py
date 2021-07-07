"""

Dataclass for any sort of modeling

@kylel

"""

from typing import Union, List

from mmda.types.annotations import Annotation
from mmda.types.document import Document

class Model:
    def predict(self, document: Document, annotations: List[Annotation]) -> List[Annotation]:
        raise NotImplementedError