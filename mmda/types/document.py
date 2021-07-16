from abc import abstractmethod

from mmda.types.document_elements import *


@dataclass
class Document:
    def __init__(
        self,
        symbols: DocumentSymbols,
        images: Optional["Image"] = None,
        **annotations
    ):

        self.symbols = symbols

        self.add(**annotations)

    def _add(self, key, value):

        # TODO: In the future, we need to more complicated things in this function
        setattr(self, key, value)

    def _annotate(self, key, value):
        pass

    def add(self, **annotations):
        """Add document annotations into this document object.
        Note: in this case, the annotations are assumed to be already associated with
        the document symbols.
        """
        for key, value in annotations.items():
            self._add(key, value)

    def annotate(self, **annotations):
        """Annotate the fields for document symbols (correlating the annotations with the
        symbols) and store them into the papers.
        """
        for key, value in annotations.items():
            self._annotate(key, value)

    def to_json(self, fields):
        pass

    def save(self, path):
        pass

    @classmethod
    def from_json(cls, json_data):
        pass

    @classmethod
    def load(cls, path: str):
        pass

    @classmethod
    def find(self, query: Span, types: str):
        pass