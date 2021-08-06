"""


"""

# TODO[kylel] not sure this class needs to exist; seems extra boilerplate for no benefit



from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field



@dataclass
class DocumentElement:
    """DocumentElement is the base class for all children objects
    of a Document. It defines the necessary APIs for manipulating
    the children objects.
    """

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    # TODO: unclear if should be `annotations` or `annotation`
    @abstractmethod
    @classmethod
    def load(cls, field_name: str, annotations: List["Annotation"], document: Optional["Document"] = None):
        pass


@dataclass
class DocumentPageSymbols(DocumentElement):
    """Storing the symbols of a page."""

    symbols: str

    # TODO: Add support for symbol bounding boxes and style
    def __getitem__(self, key):
        return self.symbols[key]

    def to_json(self):
        return self.symbols


@dataclass
class DocumentSymbols(DocumentElement):
    """Storing the symbols of a document."""

    page_count: int
    page_symbols: List[DocumentPageSymbols] = field(default_factory=list)

    # TODO[kylel] - this is more confusing than simply treating it as list[list], like it is == `docsyms[0][2:3]`
    def __getitem__(self, indices):
        page_id, symbol_slices = indices
        assert page_id < len(self.page_symbols), "Page index out of range"
        return self.page_symbols[page_id][symbol_slices]

    def to_json(self):
        return [page_symbols.to_json() for page_symbols in self.page_symbols]

    @classmethod
    def from_json(cls, symbols_dict: List[str]) -> "DocumentSymbols":
        page_symbols = [DocumentPageSymbols(symbols=page_text) for page_text in symbols_dict]
        return cls(page_count=len(page_symbols), page_symbols=page_symbols)
