from typing import Iterable, Protocol

from mmda.types.image import PILImage
from mmda.types.nouveau.protocol import Rasterizer

try:
    import pdf2image
except ImportError:
    pass


class PDF2ImageRasterizer(Rasterizer):
    def convert(self, input_pdf_path: str, **kwargs) -> Iterable[PILImage]:
        dpi = kwargs["dpi"]
        images = pdf2image.convert_from_path(pdf_path=input_pdf_path, dpi=dpi)

        return images
