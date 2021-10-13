
from abc import abstractmethod
from typing import List, Optional, Protocol, Union

from mmda.types.document import Document
from mmda.types.image import load_pdf_images_from_path, PILImage


class Rasterizer(Protocol):
    def rasterize(self, input_pdf_path: str, dpi: int, **kwargs) -> List[PILImage]:
        """Given an input PDF return a List[Image]

        Args:
            input_pdf_path (str): Path to the input PDF to process
            dpi (int): Used for specify the resolution (or `DPI, dots per inch
                       <https://en.wikipedia.org/wiki/Dots_per_inch>`_) when loading images of
                       the pdf. Higher DPI values mean clearer images (also larger file sizes).

        Returns:
            List[Image]
        """
        images = load_pdf_images_from_path(pdf_path=input_pdf_path, dpi=dpi)
        return images

