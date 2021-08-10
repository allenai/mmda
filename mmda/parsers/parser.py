"""

Dataclass for creating token streams from a document

@kylel

"""

from typing import Optional, Union, List
from abc import abstractmethod


from mmda.types.document import Document
from mmda.types.image import load_pdf_images_from_path

class BaseParser:

    @abstractmethod
    def parse(self, input_pdf_path: str, output_json_path: Optional[str] = None,
              tempdir: Optional[str] = None, load_images: bool = False) -> Union[str, Document]:
        """This is the main entrance point for using the PDF parsers. For a
        given PDF file, this method will return a Document object.
        """

    # TODO[kylel] - this doesnt quite work; convert_from_path() doesnt use our Image w patches; it uses PIL.Image
    # TODO[kylel] - serialization?
    def load_images(self, input_pdf_path: str) -> List["PIL.Image"]:
        images = load_pdf_images_from_path(input_pdf_path, dpi=72) 
        #Though 72 is not the default dpi for pdf2image, it's commonly used by other PDF parsing systems
        return images