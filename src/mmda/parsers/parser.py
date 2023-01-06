"""

Protocol for creating token streams from a document

@kylel, shannons

"""

from abc import abstractmethod
from typing import Protocol

from mmda.types.document import Document


class Parser(Protocol):
    @abstractmethod
    def parse(self, input_pdf_path: str, **kwargs) -> Document:
        """Given an input PDF return a Document with at least symbols

        Args:
            input_pdf_path (str): Path to the input PDF to process

        Returns:
            Document: Depending on parser support at least symbols in the PDF
        """
