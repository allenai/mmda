"""

Dataclass for creating token streams from a document

@kylel

"""

from mmda.types.document import Document

class Parser:
    def parse(self, infile: str, outdir: str) -> str:
        raise NotImplementedError

    def load(self, infile: str) -> Document:
        raise NotImplementedError