"""

Extract outline (table of contents) metadata from a PDF. Potentially useful for
identifying section headers in the PDF body.

@rauthur

"""
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, Dict, List, Union

import pdfminer.pdfdocument as pd
import pdfminer.pdfpage as pp
import pdfminer.pdfparser as ps
import pdfminer.pdftypes as pt
import pdfminer.psparser as pr

from mmda.types.document import Document
from mmda.types.metadata import Metadata


class PDFMinerOutlineExtractorError(Exception):
    """Base class for outline metadata extration errors."""


class PDFUnsupportedDestination(PDFMinerOutlineExtractorError):
    """Raised when a destination pointer cannot be parsed to a location."""


class PDFNoUsableDestinations(PDFMinerOutlineExtractorError):
    """Raised when there are no usable destinations for any outline metadata."""


class PDFDestinationLocationMissing(PDFMinerOutlineExtractorError):
    """Raised when there is no destination data on a particular outline item."""


class PDFNoOutlines(PDFMinerOutlineExtractorError):
    """Raised when underlying PDF contains no outline metadata (common)."""


class PDFDestinationNotFound(PDFMinerOutlineExtractorError):
    """Raised when the underlying destination cannot be found in the PDF."""


class PDFInvalidSyntax(PDFMinerOutlineExtractorError):
    """Raised when the PDF cannot be properly parsed."""


class PDFPageMissing(PDFMinerOutlineExtractorError):
    """Raised when the destination points to a page that cannot be found."""


class PDFEncryptionError(PDFMinerOutlineExtractorError):
    """Raised for encrypted PDFs."""


class PDFRecursionError(PDFMinerOutlineExtractorError):
    """Underlying parse has cyclic reference of some sort causing recursion error."""


class PDFAttributeError(PDFMinerOutlineExtractorError):
    """Wraps an underlying AttributeError."""


class PDFTypeError(PDFMinerOutlineExtractorError):
    """Wraps an underlying TypeError."""


class PDFEOF(PDFMinerOutlineExtractorError):
    """Raised when encountering an unexpected EOF in parsing."""


class PDFPSSyntaxError(PDFMinerOutlineExtractorError):
    """Raised if the PDF syntax is invalid."""


@dataclass
class _PDFDestination:
    """Destination on the target page."""

    top: float
    left: float


@dataclass
class _PDFPageInfo:
    """Enumeration of PDF page along with dimensions."""

    index: int
    x0: float
    y0: float
    x1: float
    y1: float


_OutlineItemValues = Union[str, int, float]


@dataclass
class OutlineItem:
    """A pointer to a section location in a PDF."""

    id: int  # pylint: disable=invalid-name
    title: str
    level: int
    page: int
    l: float  # pylint: disable=invalid-name
    t: float  # pylint: disable=invalid-name

    @classmethod
    def from_metadata_dict(
        cls, metadata_dict: Dict[str, _OutlineItemValues]
    ) -> "OutlineItem":
        """Instantiate from a metadata dict on a Document

        Args:
            metadata_dict (Dict[str, _OutlineItemValues]): Document metadata object

        Returns:
            OutlineItem: Rehydrated OutlineItem object
        """
        return OutlineItem(
            id=metadata_dict["id"],
            title=metadata_dict["title"],
            level=metadata_dict["level"],
            page=metadata_dict["page"],
            l=metadata_dict["l"],
            t=metadata_dict["t"],
        )

    def to_metadata_dict(self) -> Dict[str, _OutlineItemValues]:
        """Convert object to a dict for storing as metadata on Document

        Returns:
            Dict[str, _OutlineMetadataKeys]: dict representation of object
        """
        return asdict(self)


@dataclass
class Outline:
    """A class to represent an ordered list of outline items."""

    items: List[OutlineItem]

    @classmethod
    def from_metadata_dict(cls, metadata_dict: Metadata) -> "Outline":
        """Instantiate from a metadata dict on a Document

        Args:
            metadata_dict (Metadata): Document metadata object

        Returns:
            Outline: Rehydrated Outline object
        """
        return Outline(
            items=[
                OutlineItem.from_metadata_dict(i)
                for i in metadata_dict["outline"]["items"]
            ]
        )

    def to_metadata_dict(self) -> Dict[str, _OutlineItemValues]:
        """Convert object to a dict for storing as metadata on Document

        Returns:
            Dict[str, List[Dict[str, _OutlineItemValues]]]: dict representation
        """
        return asdict(self)


def _dest_to_outline_metadata(
    dest: _PDFDestination, page: int, outline_id: int, title: str, level: int
) -> OutlineItem:
    return OutlineItem(
        id=outline_id, title=title, level=level, page=page, l=dest.left, t=dest.top
    )


def _get_page_infos(doc: pd.PDFDocument) -> Dict[Any, _PDFPageInfo]:
    infos = {}

    for idx, page in enumerate(pp.PDFPage.create_pages(doc)):
        x0, y0, x1, y1 = page.mediabox
        infos[page.pageid] = _PDFPageInfo(index=idx, x0=x0, y0=y0, x1=x1, y1=y1)

    return infos


def _resolve_dest(dest, doc: pd.PDFDocument):
    if isinstance(dest, str) or isinstance(dest, bytes):
        dest = pt.resolve1(doc.get_dest(dest))
    elif isinstance(dest, pr.PSLiteral):
        dest = pt.resolve1(doc.get_dest(dest.name))
    if isinstance(dest, dict):
        dest = dest["D"]
    return dest


def _get_dest(dest: List[Any], page_info: _PDFPageInfo) -> _PDFDestination:
    w = page_info.x1 - page_info.x0
    h = page_info.y1 - page_info.y0

    if dest[1] == pr.PSLiteralTable.intern("XYZ"):
        # Sometimes the expected coordinates can be missing
        if dest[3] is None or dest[2] is None:
            raise PDFDestinationLocationMissing(f"Missing location: {dest}!")

        return _PDFDestination(top=(h - dest[3]) / h, left=dest[2] / w)

    if dest[1] == pr.PSLiteralTable.intern("FitR"):
        return _PDFDestination(top=(h - dest[5]) / h, left=dest[2] / w)
    else:
        raise PDFUnsupportedDestination(f"Unkown destination value: {dest}!")


class PDFMinerOutlineExtractor:
    """Parse a PDF and return a new Document with just outline metadata added."""

    def extract(self, input_pdf_path: str, doc: Document, **kwargs) -> Outline:
        """Get outline metadata from a PDF document and store on Document metadata.

        Args:
            input_pdf_path (str): The PDF to process
            doc (Document): The instantiated document

        Raises:
            ex: If `raise_exceptions` is passed in kwargs and evaluates as True then
                underlying processing exceptions will be raised. Otherwise an empty
                array of results will be appended to the Document.
        """
        outlines: List[OutlineItem] = []

        try:
            outlines: List[OutlineItem] = self._extract_outlines(input_pdf_path)
        except PDFMinerOutlineExtractorError as ex:
            if kwargs.get("raise_exceptions"):
                raise ex

        # If we have just one top-level outline assume that it's the title and remove
        if len([o for o in outlines if o.level == 0]) == 1:
            outlines = [
                OutlineItem(
                    id=o.id, title=o.title, level=o.level - 1, page=o.page, l=o.l, t=o.t
                )
                for o in outlines
                if o.level > 0
            ]

        return Outline(items=[o for o in outlines])

    def _extract_outlines(self, input_pdf_path: str) -> List[OutlineItem]:
        outlines = []

        with open(input_pdf_path, "rb") as pdf_file:
            pdf_bytes = BytesIO(pdf_file.read())

        try:
            psdoc = pd.PDFDocument(ps.PDFParser(pdf_bytes))
            pages = _get_page_infos(psdoc)

            # pylint: disable=invalid-name
            for oid, (level, title, dest, a, _se) in enumerate(psdoc.get_outlines()):
                page = None

                # First try to get target location from a dest object
                if dest:
                    try:
                        dest = _resolve_dest(dest, psdoc)
                        page = pages[dest[0].objid]
                    except AttributeError:
                        dest = None
                        page = None

                # If no dest, try to get it from an action object
                elif a:
                    action = a if isinstance(a, dict) else a.resolve()

                    if isinstance(action, dict):
                        subtype = action.get("S")
                        if (
                            subtype
                            and subtype == pr.PSLiteralTable.intern("GoTo")
                            and action.get("D")
                        ):
                            dest = _resolve_dest(action["D"], psdoc)
                            page = pages[dest[0].objid]

                if page is not None:
                    outlines.append(
                        _dest_to_outline_metadata(
                            dest=_get_dest(dest, page),
                            page=page.index,
                            outline_id=oid,
                            title=title,
                            level=level - 1,
                        )
                    )

            if len(outlines) == 0:
                raise PDFNoUsableDestinations("Did not find any usable dests!")

        except pd.PDFNoOutlines as ex:
            raise PDFNoOutlines() from ex
        except pd.PDFDestinationNotFound as ex:
            raise PDFDestinationNotFound() from ex
        except pd.PDFSyntaxError as ex:
            raise PDFInvalidSyntax() from ex
        except pd.PDFEncryptionError as ex:
            raise PDFEncryptionError() from ex
        except AttributeError as ex:
            raise PDFAttributeError() from ex
        except pr.PSEOF as ex:
            raise PDFEOF() from ex
        except pr.PSSyntaxError as ex:
            raise PDFPSSyntaxError() from ex
        except RecursionError as ex:
            raise PDFRecursionError() from ex
        except TypeError as ex:
            raise PDFTypeError() from ex
        except KeyError as ex:
            raise PDFPageMissing() from ex

        return outlines
