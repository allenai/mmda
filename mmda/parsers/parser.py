"""

Dataclass for creating token streams from a document

@kylel, shannons

"""

from typing import Optional, Union, List
from abc import abstractmethod

from mmda.types.document import Document
from mmda.types.image import load_pdf_images_from_path


class BaseParser:

    DEFAULT_PDF_RENDERING_DPI = 72

    def __init__(self, dpi: int = None) -> None:
        """
        dpi (int, optional):
            Used for specify the resolution (or `DPI, dots per inch
            <https://en.wikipedia.org/wiki/Dots_per_inch>`_) when loading images of
            the pdf. Higher DPI values mean clearer images (also larger file sizes).

            Defaults to `self.DEFAULT_PDF_RENDERING_DPI=72`.
        """

    @property
    def dpi(self) -> int:
        return self._dpi

    @dpi.setter
    def dpi(self, dpi: int):
        self._dpi = dpi if dpi is not None else self.DEFAULT_PDF_RENDERING_DPI

    @abstractmethod
    def parse(
        self,
        input_pdf_path: str,
        output_json_path: Optional[str] = None,
        tempdir: Optional[str] = None,
        load_images: bool = False,
    ) -> Union[str, Document]:
        """This is the main entrance point for using the PDF parsers. For a
        given PDF file, this method will return a Document object.
        """

    # TODO[kylel] - serialization?
    def load_images(self, input_pdf_path: str, dpi=None) -> List["PIL.Image"]:
        if dpi is None:
            dpi = self.dpi
        else:
            self.dpi = dpi
        images = load_pdf_images_from_path(input_pdf_path, dpi=dpi)
        # Though 72 is not the default dpi for pdf2image, it's commonly used by other PDF parsing systems
        return images