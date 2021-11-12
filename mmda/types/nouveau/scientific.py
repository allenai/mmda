from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from mmda.types.nouveau.base import BoxGroup, Document, Span, SpanGroup
from mmda.types.nouveau.protocol import Extractor


class TokenType(Enum):
    Title = 1
    Author = 2
    Abstract = 3
    Keywords = 4
    Figure = 5
    Footnote = 6
    Algorithm = 7
    List = 8
    Section = 9
    Paragraph = 10
    Equation = 11
    Bibliography = 12
    Table = 13
    Caption = 14
    Header = 15
    Footer = 16


@dataclass
class TitleGroup(SpanGroup):
    def __init__(self, id: str, spans: Iterable[Span], doc: Document):
        super().__init__(id, spans, doc)

    @property
    def text(self) -> str:
        return self._.text

    @text.setter
    def text(self, text: str):
        self._.text = text


class TitleExtractor(Extractor):
    @abstractmethod
    def extract(self, document: Document) -> TitleGroup:
        """Returns an annotated title group.

        Args:
            document (Document): The parsed document

        Returns:
            TitleGroup: Title group with text parsed from raw
        """


@dataclass
class TokenGroup(SpanGroup):
    @property
    def type(self) -> TokenType:
        return self._.type

    @type.setter
    def type(self, type: TokenType):
        self._.type = type


@dataclass
class AbstractGroup(SpanGroup):
    pass


@dataclass
class BibliographyGroup(SpanGroup):
    pass


class ResearchArticle(Document):
    """A concrete implementation of a Document offering additional typing.

    Args:
        symbols (str): Underlying document symbols
    """

    def __init__(self, symbols: str):
        super().__init__(symbols)

    @property
    def blocks(self) -> Iterable[BoxGroup]:
        return self._.blocks

    @blocks.setter
    def blocks(self, blocks: Iterable[BoxGroup]):
        self._.blocks = blocks

    @property
    def title(self) -> TitleGroup:
        return self._.title

    @title.setter
    def title(self, title: TitleGroup):
        self._.title = title

    def abstract(self) -> AbstractGroup:
        return self._.abstract

    def bibliography(self) -> Iterable[BibliographyGroup]:
        return self._.bibliography
