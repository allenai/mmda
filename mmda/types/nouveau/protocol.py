"""

@rauthur

"""

from abc import abstractmethod
from typing import Iterable, Protocol, Union

from mmda.types.image import PILImage
from mmda.types.nouveau.base import BoxGroup, Document, SpanGroup


class Parser(Protocol):
    @abstractmethod
    def parse(self, input_pdf_path: str, **kwargs) -> Document:
        """Given an input PDF return a Document with at least symbols.

        Args:
            input_pdf_path (str): Path to the input PDF to process

        Returns:
            Document: Depending on parser support at least symbols in the PDF
        """


class Rasterizer(Protocol):
    @abstractmethod
    def convert(self, input_pdf_path: str, **kwargs) -> Iterable[PILImage]:
        """Given an input PDF return an iterable of page images in order.

        Args:
            input_pdf_path (str): Path to the input PDF to process

        Returns:
            Iterable[PILImage]: Iterable of rendered page images
        """


class BoxPredictor(Protocol):
    @abstractmethod
    def predict(self, document: Document) -> Iterable[BoxGroup]:
        """Given a document produce an iterable of visual boxes.

        Args:
            document (Document): The parsed and rasterized document

        Returns:
            Iterable[BoxGroup]: Iterable of visual boxes
        """


class SpanPredictor(Protocol):
    @abstractmethod
    def predict(self, document: Document) -> Iterable[SpanGroup]:
        """Given a document produce an iterable of spans with custom annotations.

        Args:
            document (Document): The parsed document

        Returns:
            Iterable[SpanGroup]: Iterable of span groups with any custom annotations
        """


SpanGroupOrGroups = Union[SpanGroup, Iterable[SpanGroup]]


class Extractor(Protocol):
    @abstractmethod
    def extract(self, span_group: SpanGroupOrGroups) -> SpanGroupOrGroups:
        """Extract information from an annotated span group(s).

        Args:
            span_group (Union[SpanGroup, Iterable[SpanGroup]]): Annotated groups

        Returns:
            Union[SpanGroup, Iterable[SpanGroup]]: A span group with any additional
                custom annotations
        """
