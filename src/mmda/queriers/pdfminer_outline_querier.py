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
from mmda.queriers.querier import Querier
from mmda.types.document import Document


class PDFMinerOutlineParserError(Exception):
    """Base class for outline metadata extration errors."""


class PDFUnsupportedDestination(PDFMinerOutlineParserError):
    """Raised when a destination pointer cannot be parsed to a location."""


class PDFNoUsableDestinations(PDFMinerOutlineParserError):
    """Raised when there are no usable destinations for any outline metadata."""


class PDFDestinationLocationMissing(PDFMinerOutlineParserError):
    """Raised when there is no destination data on a particular outline item."""


class PDFNoOutlines(PDFMinerOutlineParserError):
    """Raised when underlying PDF contains no outline metadata (common)."""


class PDFDestinationNotFound(PDFMinerOutlineParserError):
    """Raised when the underlying destination cannot be found in the PDF."""


class PDFInvalidSyntax(PDFMinerOutlineParserError):
    """Raised when the PDF cannot be properly parsed."""


class PDFPageMissing(PDFMinerOutlineParserError):
    """Raised when the destination points to a page that cannot be found."""


class PDFEncryptionError(PDFMinerOutlineParserError):
    """Raised for encrypted PDFs."""


class PDFRecursionError(PDFMinerOutlineParserError):
    """Underlying parse has cyclic reference of some sort causing recursion error."""


class PDFAttributeError(PDFMinerOutlineParserError):
    """Wraps an underlying AttributeError."""


class PDFTypeError(PDFMinerOutlineParserError):
    """Wraps an underlying TypeError."""


class PDFEOF(PDFMinerOutlineParserError):
    """Raised when encountering an unexpected EOF in parsing."""


class PDFPSSyntaxError(PDFMinerOutlineParserError):
    """Raised if the PDF syntax is invalid."""


@dataclass
class _PDFDestination:
    """Destination on the target page."""

    top: float
    left: float


_OutlineMetadataKeys = Union[str, int, float]


@dataclass
class OutlineMetadata:
    """A pointer to a section location in a PDF."""

    id: int  # pylint: disable=invalid-name
    title: str
    level: int
    page: int
    l: float  # pylint: disable=invalid-name
    t: float  # pylint: disable=invalid-name

    @classmethod
    def from_metadata_dict(
        cls, metadata_dict: Dict[str, _OutlineMetadataKeys]
    ) -> "OutlineMetadata":
        """Instantiate from a metadata dict on a Document

        Args:
            metadata_dict (Dict[str, _OutlineMetadataKeys]): Document metadata object

        Returns:
            OutlineMetadata: Rehydrated OutlineMetadata object
        """
        return OutlineMetadata(
            id=metadata_dict["id"],
            title=metadata_dict["title"],
            level=metadata_dict["level"],
            page=metadata_dict["page"],
            l=metadata_dict["l"],
            t=metadata_dict["t"],
        )

    def to_metadata_dict(self) -> Dict[str, _OutlineMetadataKeys]:
        """Convert object to a dict for storing as metadata on Document

        Returns:
            Dict[str, _OutlineMetadataKeys]: dict representation of object
        """
        return asdict(self)


def _dest_to_outline_metadata(
    dest: _PDFDestination, page: int, outline_id: int, title: str, level: int
) -> OutlineMetadata:
    return OutlineMetadata(
        id=outline_id, title=title, level=level, page=page, l=dest.left, t=dest.top
    )


def _get_pages(doc: pd.PDFDocument):
    return {p.pageid: i for i, p in enumerate(pp.PDFPage.create_pages(doc))}


def _resolve_dest(dest, doc: pd.PDFDocument):
    if isinstance(dest, str) or isinstance(dest, bytes):
        dest = pt.resolve1(doc.get_dest(dest))
    elif isinstance(dest, pr.PSLiteral):
        dest = pt.resolve1(doc.get_dest(dest.name))
    if isinstance(dest, dict):
        dest = dest["D"]
    return dest


def _get_dest(dest: List[Any]) -> _PDFDestination:
    if dest[1] == pr.PSLiteralTable.intern("XYZ"):
        # Sometimes the expected coordinates can be missing
        if dest[3] is None or dest[2] is None:
            raise PDFDestinationLocationMissing(f"Missing location: {dest}!")

        return _PDFDestination(top=dest[3], left=dest[2])

    if dest[1] == pr.PSLiteralTable.intern("FitR"):
        return _PDFDestination(top=dest[5], left=dest[2])
    else:
        raise PDFUnsupportedDestination(f"Unkown destination value: {dest}!")


class PDFMinerOutlineQuerier(Querier):
    """Parse a PDF and return a new Document with just outline metadata added."""

    def query(self, input_pdf_path: str, doc: Document, **kwargs) -> None:
        """Query outline metadata from a PDF document and store on Document metadata.

        Args:
            input_pdf_path (str): The PDF to process
            doc (Document): The instantiated document

        Raises:
            ex: If `raise_exceptions` is passed in kwargs and evaluates as True then
                underlying processing exceptions will be raised. Otherwise an empty
                array of results will be appended to the Document.
        """
        outlines: List[OutlineMetadata] = []

        try:
            outlines: List[OutlineMetadata] = self._extract_outlines(input_pdf_path)
        except PDFMinerOutlineParserError as ex:
            if kwargs.get("raise_exceptions"):
                raise ex

        # If we have just one top-level outline assume that it's the title and remove
        if len([o for o in outlines if o.level == 0]) == 1:
            outlines = [
                OutlineMetadata(
                    id=o.id, title=o.title, level=o.level - 1, page=o.page, l=o.l, t=o.t
                )
                for o in outlines
                if o.level > 0
            ]

        doc.add_metadata(outlines=[o.to_metadata_dict() for o in outlines])

    def _extract_outlines(self, input_pdf_path: str) -> List[OutlineMetadata]:
        outlines = []

        with open(input_pdf_path, "rb") as pdf_file:
            pdf_bytes = BytesIO(pdf_file.read())

        try:
            psdoc = pd.PDFDocument(ps.PDFParser(pdf_bytes))
            pages = _get_pages(psdoc)

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
                            dest=_get_dest(dest),
                            page=page,
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
