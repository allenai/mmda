"""

Dataclass for creating token streams from a document

@kyle

"""


class Parser:
    def __init__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class TextParser(Parser):
    pass


class VisionParser(Parser):
    pass