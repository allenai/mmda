"""


@kylel

"""

from mmda.types import *

from abc import abstractmethod


class Recipe:
    @abstractmethod
    def from_path(self, pdfpath: str) -> Document:
        raise NotImplementedError

    @abstractmethod
    def from_doc(self, doc: Document) -> Document:
        raise NotImplementedError
