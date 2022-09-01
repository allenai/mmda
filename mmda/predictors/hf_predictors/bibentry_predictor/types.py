from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from mmda.types.annotation import SpanGroup


class BibEntryLabel(Enum):
    MISC = 0

    CITATION_NUMBER = 1

    AUTHOR_START = 2
    AUTHOR_MIDDLE = 3
    AUTHOR_END = 4

    ISSUED_DAY = 5
    ISSUED_MONTH = 6
    ISSUED_YEAR = 7

    TITLE = 8
    JOURNAL = 9
    PUBLISHER = 10
    VOLUME = 11
    ISSUE = 12
    PAGE = 13
    URL = 14
    DOI = 15
    EVENT = 16
    ISBN = 17


class StringWithSpan(BaseModel):
    content: str
    start: int  # inclusive
    end: int  # exclusive


class BibEntryPredictionWithSpan(BaseModel):
    citation_number: Optional[StringWithSpan]
    authors: Optional[List[StringWithSpan]]
    title: Optional[StringWithSpan]
    journal_venue_or_event: Optional[StringWithSpan]
    year: Optional[StringWithSpan]
    doi: Optional[StringWithSpan]
    url: Optional[StringWithSpan]


@dataclass
class BibEntryStructureSpanGroups:
    bib_entry_number: List[SpanGroup] = field(default_factory=list)
    bib_entry_authors: List[SpanGroup] = field(default_factory=list)
    bib_entry_title: List[SpanGroup] = field(default_factory=list)
    bib_entry_venue_or_event: List[SpanGroup] = field(default_factory=list)
    bib_entry_year: List[SpanGroup] = field(default_factory=list)
    bib_entry_doi: List[SpanGroup] = field(default_factory=list)
    bib_entry_url: List[SpanGroup] = field(default_factory=list)
