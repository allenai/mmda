from contextlib import ExitStack
import itertools
import string
from typing import List, Optional, Union

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError

try:
    # pdfplumber >= 0.8.0
    import pdfplumber.utils.text as ppu
except:
    # pdfplumber <= 0.7.6
    import pdfplumber.utils as ppu

from mmda.parsers.parser import Parser
from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.metadata import Metadata
from mmda.types.names import PagesField, RowsField, SymbolsField, TokensField
from mmda.types.span import Span

_TOL = Union[int, float]


class PdfParsingError(PDFSyntaxError):
    ...


class WordExtractorWithFontInfo(ppu.WordExtractor):
    """Override WordExtractor methods to append additional char-level info."""

    def __init__(
        self,
        x_tolerance: _TOL = ppu.DEFAULT_X_TOLERANCE,
        y_tolerance: _TOL = ppu.DEFAULT_Y_TOLERANCE,
        keep_blank_chars: bool = False,
        use_text_flow: bool = False,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
        split_at_punctuation: Union[bool, str] = False,
        append_attrs: Optional[List[str]] = None,
    ):
        super().__init__(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=keep_blank_chars,
            use_text_flow=use_text_flow,
            horizontal_ltr=horizontal_ltr,
            vertical_ttb=vertical_ttb,
            extra_attrs=extra_attrs,
            split_at_punctuation=split_at_punctuation,
        )
        self.append_attrs = append_attrs

    def merge_chars(self, ordered_chars: ppu.T_obj_list) -> ppu.T_obj:
        """Modify returned word to capture additional char-level info."""
        word = super().merge_chars(ordered_chars=ordered_chars)

        # ordered_chars is a list of characters for the word with extra attributes.
        # Here we simply append additional information to the word using the first char.
        for key in self.append_attrs:
            if key not in word:
                word[key] = ordered_chars[0][key]

        return word

    def extract(self, chars: ppu.T_obj_list) -> ppu.T_obj_list:
        if hasattr(super(), 'extract'):
            # pdfplumber <= 0.7.6
            return super().extract(chars)
        else:
            # pdfplumber >= 0.8.0
            return super().extract_words(chars)


class PDFPlumberParser(Parser):
    # manually added characters: '–' and '§'
    DEFAULT_PUNCTUATION_CHARS = string.punctuation + chr(8211) + chr(167)

    def __init__(
        self,
        token_x_tolerance: Union[int, float] = 1.5,
        token_y_tolerance: Union[int, float] = 2,
        line_x_tolerance: Union[int, float] = 10,
        line_y_tolerance: Union[int, float] = 10,
        keep_blank_chars: bool = False,
        use_text_flow: bool = True,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
        split_at_punctuation: Union[str, bool] = True,
    ):
        """The PDFPlumber PDF Detector
        Args:
            token_x_tolerance (int, optional):
                The threshold used for extracting "word tokens" from the pdf file.
                It will merge the pdf characters into a word token if the difference
                between the x_2 of one character and the x_1 of the next is less than
                or equal to token_x_tolerance. See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to 1.5, in absolute coordinates.
            token_y_tolerance (int, optional):
                The threshold used for extracting "word tokens" from the pdf file.
                It will merge the pdf characters into a word token if the difference
                between the y_2 of one character and the y_1 of the next is less than
                or equal to token_y_tolerance. See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to 2, in absolute coordinates.
            line_x_tolerance (int, optional):
                The threshold used for extracting "line tokens" from the pdf file.
                Defaults to 10, in absolute coordinates.
            line_y_tolerance (int, optional):
                The threshold used for extracting "line tokens" from the pdf file.
                Defaults to 10, in absolute coordinates.
            keep_blank_chars (bool, optional):
                When keep_blank_chars is set to True, it will treat blank characters
                are treated as part of a word, not as a space between words. See
                details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to False.
            use_text_flow (bool, optional):
                When use_text_flow is set to True, it will use the PDF's underlying
                flow of characters as a guide for ordering and segmenting the words,
                rather than presorting the characters by x/y position. (This mimics
                how dragging a cursor highlights text in a PDF; as with that, the
                order does not always appear to be logical.) See details in
                `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to True.
            horizontal_ltr (bool, optional):
                When horizontal_ltr is set to True, it means the doc should read
                text from left to right, vice versa.
                Defaults to True.
            vertical_ttb (bool, optional):
                When vertical_ttb is set to True, it means the doc should read
                text from top to bottom, vice versa.
                Defaults to True.
            extra_attrs (Optional[List[str]], optional):
                Passing a list of extra_attrs (e.g., ["fontname", "size"]) will
                restrict each words to characters that share exactly the same
                value for each of those `attributes extracted by pdfplumber
                <https://github.com/jsvine/pdfplumber/blob/develop/README.md#char-properties>`_,
                and the resulting word dicts will indicate those attributes.
                See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to `["fontname", "size"]`.
        """
        self.token_x_tolerance = token_x_tolerance
        self.token_y_tolerance = token_y_tolerance
        self.line_x_tolerance = line_x_tolerance
        self.line_y_tolerance = line_y_tolerance
        self.keep_blank_chars = keep_blank_chars
        self.use_text_flow = use_text_flow
        self.horizontal_ltr = horizontal_ltr
        self.vertical_ttb = vertical_ttb
        self.extra_attrs = (
            extra_attrs if extra_attrs is not None else ["fontname", "size"]
        )
        if split_at_punctuation is True:
            split_at_punctuation = type(self).DEFAULT_PUNCTUATION_CHARS
        self.split_at_punctuation = split_at_punctuation

    def parse(self, input_pdf_path: str, **_) -> Document:
        """Parse the pdf file and return a Document object.

        Args:
            input_pdf_path (str): The path to the pdf file.
        """

        # using a stack so we can detect if the open() fails
        with ExitStack() as stack:

            try:
                plumber_pdf_object = stack.enter_context(
                    pdfplumber.open(input_pdf_path)
                )
            except PDFSyntaxError as e:
                raise PdfParsingError(
                    f"Failed to open '{input_pdf_path}' with pdfplumber. "
                    f"Please check if the file is a valid pdf file. "
                ) from e

            all_tokens = []
            all_word_ids = []
            last_word_id = -1
            all_row_ids = []
            last_row_id = -1
            all_page_ids = []
            for page_id, page in enumerate(plumber_pdf_object.pages):
                # 1) tokens we use for Document.symbols
                coarse_tokens = page.extract_words(
                    x_tolerance=self.token_x_tolerance,
                    y_tolerance=self.token_y_tolerance,
                    keep_blank_chars=self.keep_blank_chars,
                    use_text_flow=self.use_text_flow,
                    horizontal_ltr=self.horizontal_ltr,
                    vertical_ttb=self.vertical_ttb,
                    extra_attrs=self.extra_attrs,
                    split_at_punctuation=None,
                )
                # 2) tokens we use for Document.tokens
                fine_tokens = WordExtractorWithFontInfo(
                    x_tolerance=self.token_x_tolerance,
                    y_tolerance=self.token_y_tolerance,
                    keep_blank_chars=self.keep_blank_chars,
                    use_text_flow=self.use_text_flow,
                    horizontal_ltr=self.horizontal_ltr,
                    vertical_ttb=self.vertical_ttb,
                    extra_attrs=self.extra_attrs,
                    split_at_punctuation=self.split_at_punctuation,
                    append_attrs=["fontname", "size"],
                ).extract(page.chars)
                # 3) align fine tokens with coarse tokens
                word_ids_of_fine_tokens = self._align_coarse_and_fine_tokens(
                    coarse_tokens=[c["text"] for c in coarse_tokens],
                    fine_tokens=[f["text"] for f in fine_tokens],
                )
                if len(word_ids_of_fine_tokens) != len(fine_tokens):
                    raise ValueError(
                        "The length of word_ids_of_fine_tokens and fine_tokens "
                        f"should be the same on page {page_id}, but got "
                        f"{len(word_ids_of_fine_tokens)} and {len(fine_tokens)}."
                        "Please report this issue on Github, and include the "
                        "pdf file that caused this error."
                    )

                # 4) normalize / clean tokens & boxes
                if len(fine_tokens) > 0:
                    fine_tokens = [
                        {
                            "text": token["text"],
                            "fontname": token["fontname"],
                            "size": token["size"],
                            "bbox": Box.from_pdf_coordinates(
                                x1=float(token["x0"]),
                                y1=float(token["top"]),
                                x2=float(token["x1"]),
                                y2=float(token["bottom"]),
                                page_width=float(page.width),
                                page_height=float(page.height),
                                page=int(page_id),
                            ).get_relative(
                                page_width=float(page.width),
                                page_height=float(page.height)
                            ),
                        }
                        for token in fine_tokens
                    ]
                else:
                    # this page does not have tokens, so we add a single
                    # fictitious token to the page. This is to ensure that
                    # any element that is on this page gets anchored to
                    # something
                    fine_tokens = [
                        {
                            # adding one single character: the null character
                            "text": "\u0000",
                            "fontname": "",
                            "size": 0.,
                            "bbox": Box.from_pdf_coordinates(
                                x1=0.,
                                y1=0.,
                                x2=1.,
                                y2=1.,
                                page_width=float(page.width),
                                page_height=float(page.height),
                                page=int(page_id),
                            ).get_relative(
                                page_width=float(page.width),
                                page_height=float(page.height)
                            ),
                        }
                    ]
                    word_ids_of_fine_tokens = [0]

                # 5) group tokens into lines
                # TODO - doesnt belong in parser; should be own predictor
                line_ids_of_fine_tokens = self._simple_line_detection(
                    page_tokens=fine_tokens,
                    x_tolerance=self.line_x_tolerance / page.width,
                    y_tolerance=self.line_y_tolerance / page.height,
                )
                if len(line_ids_of_fine_tokens) != len(fine_tokens):
                    raise ValueError(
                        "The length of line_ids_of_fine_tokens and fine_tokens "
                        f"should be the same on page {page_id}, but got "
                        f"{len(line_ids_of_fine_tokens)} and {len(fine_tokens)}."
                        "Please report this issue on Github, and include the "
                        "pdf file that caused this error."
                    )

                # 6) accumulate
                all_tokens.extend(fine_tokens)
                all_row_ids.extend(
                    [i + last_row_id + 1 for i in line_ids_of_fine_tokens]
                )

                # 7) Update the last word id and row id for this page
                last_row_id = all_row_ids[-1]
                all_word_ids.extend(
                    [i + last_word_id + 1 for i in word_ids_of_fine_tokens]
                )
                last_word_id = all_word_ids[-1]

                for _ in fine_tokens:
                    all_page_ids.append(page_id)

            # now turn into a beautiful document!
            doc_json = self._convert_nested_text_to_doc_json(
                token_dicts=all_tokens,
                word_ids=all_word_ids,
                row_ids=all_row_ids,
                page_ids=all_page_ids,
            )

            doc = Document.from_json(doc_json)
            return doc

    def _convert_nested_text_to_doc_json(
        self,
        token_dicts: List[dict],
        word_ids: List[int],
        row_ids: List[int],
        page_ids: List[int],
    ) -> dict:
        """For a single page worth of text"""

        # 1) build tokens & symbols
        symbols = ""
        token_annos: List[SpanGroup] = []
        start = 0
        for token_id in range(len(token_dicts) - 1):

            token_dict = token_dicts[token_id]
            current_word_id = word_ids[token_id]
            next_word_id = word_ids[token_id + 1]
            current_row_id = row_ids[token_id]
            next_row_id = row_ids[token_id + 1]

            # 1) add to symbols
            symbols += token_dict["text"]

            # 2) make Token
            end = start + len(token_dict["text"])

            # 2b) gather token metadata
            if "fontname" in token_dict and "size" in token_dict:
                token_metadata = Metadata(
                    fontname=token_dict["fontname"],
                    size=token_dict["size"],
                )
            else:
                token_metadata = None

            token = SpanGroup(
                spans=[Span(start=start, end=end, box=token_dict["bbox"])],
                id=token_id,
                metadata=token_metadata,
            )
            token_annos.append(token)

            # 3) increment whitespace based on Row & Word membership. and build Rows.
            if next_row_id == current_row_id:
                if next_word_id == current_word_id:
                    start = end
                else:
                    symbols += " "
                    start = end + 1
            else:
                # new row
                symbols += "\n"
                start = end + 1
        # handle last token if we have one
        if len(token_dicts) > 0:
            symbols += token_dicts[-1]["text"]
            end = start + len(token_dicts[-1]["text"])
            last_token = SpanGroup(
                spans=[Span(start=start, end=end, box=token_dicts[-1]["bbox"])],
                id=len(token_dicts) - 1,
            )
            token_annos.append(last_token)

        # 2) build rows
        tokens_with_group_ids = [
            (token, row_id, page_id)
            for token, row_id, page_id in zip(token_annos, row_ids, page_ids)
        ]
        row_annos: List[SpanGroup] = []
        for row_id, tups in itertools.groupby(
            iterable=tokens_with_group_ids, key=lambda tup: tup[1]
        ):
            row_tokens = [token for token, _, _ in tups]
            row = SpanGroup(
                spans=[
                    Span(
                        start=row_tokens[0][0].start,
                        end=row_tokens[-1][0].end,
                        box=Box.small_boxes_to_big_box(
                            boxes=[span.box for t in row_tokens for span in t]
                        ),
                    )
                ],
                id=row_id,
            )
            row_annos.append(row)

        # 3) build pages
        page_annos: List[SpanGroup] = []
        for page_id, tups in itertools.groupby(
            iterable=tokens_with_group_ids, key=lambda tup: tup[2]
        ):
            page_tokens = [token for token, _, _ in tups]
            page = SpanGroup(
                spans=[
                    Span(
                        start=page_tokens[0][0].start,
                        end=page_tokens[-1][0].end,
                        box=Box.small_boxes_to_big_box(
                            boxes=[span.box for t in page_tokens for span in t]
                        ),
                    )
                ],
                id=page_id,
            )
            page_annos.append(page)

        return {
            SymbolsField: symbols,
            TokensField: [token.to_json() for token in token_annos],
            RowsField: [row.to_json() for row in row_annos],
            PagesField: [page.to_json() for page in page_annos],
        }

    def _simple_line_detection(
        self,
        page_tokens: List[dict],
        x_tolerance: Union[int, float] = 10,
        y_tolerance: Union[int, float] = 10
    ) -> List[int]:
        """Get text lines from the page_tokens.
        It will automatically add new lines for 1) line breaks (i.e., the current token
        has a larger y_difference between the previous one than the y_tolerance) or
        2) big horizontal gaps (i.e., the current token has a larger y_difference between
        the previous one than the x_tolerance)

        Adapted from https://github.com/allenai/VILA/blob/e6d16afbd1832f44a430074855fbb4c3d3604f4a/src/vila/pdftools/pdfplumber_extractor.py#L24

        Modified Oct 2022 (kylel): Changed return value to be List[int]
        """
        prev_y = None
        prev_x = None

        lines = []
        cur_line_id = 0
        n = 0

        for token in page_tokens:
            cur_y = token["bbox"].center[1]
            cur_x = token["bbox"].coordinates[0]

            if prev_y is None:
                prev_y = cur_y
                prev_x = cur_x

            if abs(cur_y - prev_y) <= y_tolerance and cur_x - prev_x <= x_tolerance:

                lines.append(cur_line_id)
                if n == 0:
                    prev_y = cur_y
                else:
                    prev_y = (prev_y * n + cur_y) / (n + 1)  # EMA of the y_height
                n += 1

            else:

                cur_line_id += 1

                lines.append(cur_line_id)
                n = 1
                prev_y = cur_y

            prev_x = token["bbox"].coordinates[2]

        return lines

    def _align_coarse_and_fine_tokens(
        self, coarse_tokens: List[str], fine_tokens: List[str]
    ) -> List[int]:
        """Returns a list of length len(fine_tokens) where elements of the list are
        integer indices into coarse_tokens elements."""
        assert len(coarse_tokens) <= len(
            fine_tokens
        ), f"This method requires |coarse| <= |fine|"
        assert "".join(coarse_tokens) == "".join(
            fine_tokens
        ), f"This method requires the chars(coarse) == chars(fine)"

        coarse_start_ends = []
        start = 0
        for token in coarse_tokens:
            end = start + len(token)
            coarse_start_ends.append((start, end))
            start = end

        fine_start_ends = []
        start = 0
        for token in fine_tokens:
            end = start + len(token)
            fine_start_ends.append((start, end))
            start = end

        fine_id = 0
        coarse_id = 0
        out = []
        while fine_id < len(fine_start_ends) and coarse_id < len(coarse_start_ends):
            fine_start, fine_end = fine_start_ends[fine_id]
            coarse_start, coarse_end = coarse_start_ends[coarse_id]
            if coarse_start <= fine_start and fine_end <= coarse_end:
                out.append(coarse_id)
                fine_id += 1
            else:
                coarse_id += 1

        return out
