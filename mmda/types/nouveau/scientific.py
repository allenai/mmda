import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from mmda.predictors.heuristic_predictors.grobid_citation_predictor import get_title
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
    Unknown = 99


@dataclass
class TokenGroup(SpanGroup):
    def __init__(self, spans: Iterable[Span]):
        super().__init__(spans)

    @property
    def type(self) -> TokenType:
        return self._.type

    @type.setter
    def type(self, type: TokenType):
        self._.type = type


@dataclass
class TitleGroup(SpanGroup):
    def __init__(self, spans: Iterable[Span]):
        super().__init__(spans)

    @classmethod
    def null_instance(cls):
        title_group = cls(spans=[])
        title_group.text = None

        return title_group

    @property
    def text(self) -> Optional[str]:
        return self._.text

    @text.setter
    def text(self, text: str):
        self._.text = text


@dataclass
class AbstractGroup(SpanGroup):
    pass


@dataclass
class AuthorGroup(SpanGroup):
    pass


@dataclass
class BibliographyGroup(SpanGroup):
    def __init__(self, spans: Iterable[Span]):
        super().__init__(spans)

    @classmethod
    def null_instance(cls):
        bib_group = cls(spans=[])

        return bib_group

    @property
    def text(self) -> str:
        return self._.text

    @text.setter
    def text(self, value: str):
        self._.text = value

    @property
    def title(self) -> str:
        return self._.title

    @title.setter
    def title(self, value: str):
        self._.title = value


class TokenGroupTitleExtractor(Extractor):
    """Prioritizes the first token group tagged as title."""

    def extract(self, document: Document) -> TitleGroup:
        # Custom field with TokenGroup
        preds: Iterable[TokenGroup] = document._.preds
        token_groups = [x for x in preds if x.type == TokenType.Title]

        if len(token_groups) == 0:
            return TitleGroup.null_instance()

        title_texts = token_groups[0].symbols
        title_spans = list(itertools.chain(*[x.spans for x in token_groups]))
        title_group = TitleGroup(spans=title_spans)
        title_group.text = " ".join(title_texts)

        return title_group


class TokenGroupBibliographyExtractor(Extractor):
    """Grab all the entries around bibliographies and parse with Grobid."""

    def extract(self, document: Document) -> Iterable[BibliographyGroup]:
        preds: Iterable[TokenGroup] = document._.preds
        token_groups = [x for x in preds if x.type == TokenType.Bibliography]

        if len(token_groups) == 0:
            return BibliographyGroup.null_instance()

        bib_spans = [x.spans for x in token_groups]
        bib_groups = [BibliographyGroup(spans=spans) for spans in bib_spans]

        for bib_group in bib_groups:
            bib_group.text = document.symbols_for(bib_group)[0]
            bib_group.title = get_title(bib_group.text)

        return bib_groups


class ResearchArticle(Document):
    """A concrete implementation of a Document offering additional typing.

    Args:
        symbols (str): Underlying document symbols
    """

    def __init__(self, symbols: str):
        super().__init__(symbols)

    @classmethod
    def from_document(cls, document: Document) -> "ResearchArticle":
        article = ResearchArticle(document.symbols)
        article._.pages = document.pages
        article._.tokens = document.tokens
        article._.rows = document.rows

        return article

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

    @property
    def abstract(self) -> AbstractGroup:
        return self._.abstract

    @abstract.setter
    def abstract(self, abstract: AbstractGroup):
        self._.abstract = abstract

    @property
    def bibliography(self) -> Iterable[BibliographyGroup]:
        return self._.bibliography

    @bibliography.setter
    def bibliography(self, bibliography: BibliographyGroup):
        self._.bibliography = bibliography
