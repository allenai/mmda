"""

Dataclass for creating token streams from a document

@kylel

"""

from typing import Optional, Union

from mmda.types.document import Document

class Parser:
    def parse(self, infile: str, outdir: Optional[str] = None, outfname: Optional[str] = None) -> Union[str, Document]:
        raise NotImplementedError

    def load(self, infile: str) -> Document:
        raise NotImplementedError