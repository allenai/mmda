"""

Dataclass for creating token streams from a document

@kylel, shannons

"""

from abc import abstractmethod
from typing import List, Optional, Protocol, Union

from mmda.types.document import Document


class Parser(Protocol):
    @abstractmethod
    def parse(self,
              input_pdf_path: str,
              output_json_path: Optional[str] = None,
              **kwargs) -> Document:
        """Given an input PDF return a Document with at least symbols

        Args:
            input_pdf_path (str): Path to the input PDF to process
            output_json_path (str):  Optional path to the output JSON file

        Returns:
            Document: Depending on parser support at least symbols in the PDF
        """
