from abc import abstractmethod

from mmda.types.document_elements import *


@dataclass
class Document:

    DEFAULT_FIELDS = ["symbols", "images"]

    def __init__(
        self,
        symbols: DocumentSymbols,
        images: Optional["Image.Image"] = None,
    ):

        self.symbols = symbols
        self.images = images
        self._fields = self.DEFAULT_FIELDS

    @property
    def fields(self):  
        return self._fields

    def _check_valid_field_name(self, field_name):
        assert not field_name.startswith("_"), "The field_name should not start with `_`. "
        assert field_name not in ["fields"], "The field_name should not be 'fields'."
        assert field_name not in dir(self), f"The field_name should not conflict with existing class properties {field_name}"

    def _register_field(self, field_name):
        if field_name not in self.fields:
            self._check_valid_field_name(field_name)
            self._fields.append(field)

    def _add(self, name, annotation):

        assert annotation.doc == self # check that the annotation is associated with the document
        setattr(self, name, annotation)
        self._register_field(name)

    def _annotate(self, name, annotation):
        
        annotation = annotation.annotate(self)
        self._add(name, annotation)

    def add(self, **annotations):
        """Add document annotations into this document object.
        Note: in this case, the annotations are assumed to be already associated with
        the document symbols.
        """
        for name, annotation in annotations.items():
            self._add(name, annotation)

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
    def find(self, query: Span, field: str):
        pass