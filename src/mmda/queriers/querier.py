"""

A Querier accepts both the original PDF and the created Document and appends metadata.

@rauthur

"""

from abc import abstractmethod
from typing import Protocol

from mmda.types.document import Document


class Querier(Protocol):
    """Protocol definition for querier."""

    @abstractmethod
    def query(self, input_pdf_path: str, doc: Document, **kwargs) -> None:
        """Attempts to extract information from PDF and apply as document metadata.

        Args:
            input_pdf_path (str): Path to the source PDF
            doc (Document): Parser output for PDF
        """
