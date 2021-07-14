"""

Dataclass for creating token streams from a document

@kylel

"""

from typing import Optional, Union, List

from pdf2image import convert_from_path

from mmda.types.document import Document
from mmda.types.image import Image


class Parser:
    def parse(
        self, infile: str, outdir: Optional[str] = None, outfname: Optional[str] = None
    ) -> Union[str, Document]:
        raise NotImplementedError

    def load(self, infile: str) -> Document:
        raise NotImplementedError

    def load_images(self, infile: str) -> List["PIL.Image"]:

        images = convert_from_path(infile, dpi=72) #Though 72 is not the default dpi for pdf2image, it's commonly used by other PDF parsing systems 
        return images