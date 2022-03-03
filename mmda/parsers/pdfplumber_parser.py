from typing import Optional, List, Dict, Tuple, Union

import os
import json
import pdfplumber
import pandas as pd

from mmda.types.span import Span
from mmda.types.box import Box
from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.parsers.parser import Parser
from mmda.types.names import *


def simple_line_detection(
    page_tokens: List[Dict], x_tolerance: int = 10, y_tolerance: int = 10
) -> Dict[int, List]:
    """Get text lines from the page_tokens.
    It will automatically add new lines for 1) line breaks (i.e., the current token
    has a larger y_difference between the previous one than the y_tolerance) or
    2) big horizontal gaps (i.e., the current token has a larger y_difference between
    the previous one than the x_tolerance)

    Adapted from https://github.com/allenai/VILA/blob/e6d16afbd1832f44a430074855fbb4c3d3604f4a/src/vila/pdftools/pdfplumber_extractor.py#L24
    """
    prev_y = None
    prev_x = None

    lines = dict()
    cur_line_id = 0
    token_in_this_line = []
    n = 0

    for token in page_tokens:
        cur_y = token["bbox"].center[1]
        cur_x = token["bbox"].coordinates[0]

        if prev_y is None:
            prev_y = cur_y
            prev_x = cur_x

        if abs(cur_y - prev_y) <= y_tolerance and cur_x - prev_x <= x_tolerance:

            token_in_this_line.append(token)
            if n == 0:
                prev_y = cur_y
            else:
                prev_y = (prev_y * n + cur_y) / (n + 1)  # EMA of the y_height
            n += 1

        else:

            lines[cur_line_id] = token_in_this_line
            cur_line_id += 1

            token_in_this_line = [token]
            n = 1
            prev_y = cur_y

        prev_x = token["bbox"].coordinates[2]

    if token_in_this_line:
        lines[cur_line_id] = token_in_this_line

    return lines


class PDFPlumberParser(Parser):
    def __init__(
        self,
        token_x_tolerance: int = 1.5,
        token_y_tolerance: int = 2,
        line_x_tolerance: int = 10,
        line_y_tolerance: int = 10,
        keep_blank_chars: bool = False,
        use_text_flow: bool = True,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
        dpi=None,
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
            dpi (int, optional):
                Used for specify the resolution (or `DPI, dots per inch
                <https://en.wikipedia.org/wiki/Dots_per_inch>`_) when loading images of
                the pdf. Higher DPI values mean clearer images (also larger file sizes).

                Defaults to `self.DEFAULT_PDF_RENDERING_DPI=72`.
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
        self.dpi = dpi

    def parse(self, input_pdf_path: str) -> Document:
        doc = self._load_pdf_as_doc(input_pdf_path)
        return doc

    def _load_page_tokens(
        self,
        page: pdfplumber.page.Page,
        page_index: int,
        x_tolerance: int = 1.5,
        y_tolerance: int = 2,
        keep_blank_chars: bool = False,
        use_text_flow: bool = True,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
    ):
        token_data = page.extract_words(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=keep_blank_chars,
            use_text_flow=use_text_flow,
            horizontal_ltr=horizontal_ltr,
            vertical_ttb=vertical_ttb,
            extra_attrs=extra_attrs,
        )

        page_tokens = [
            {
                "text": token["text"],
                "bbox": Box.from_coordinates(
                    x1=float(token["x0"]),
                    y1=float(token["top"]),
                    x2=float(token["x1"]),
                    y2=float(token["bottom"]),
                    page=page_index,
                ).get_relative(
                    page_width=float(page.width), page_height=float(page.height)
                ),
            }
            for token in token_data
        ]

        return page_tokens

    def _load_pdf_tokens(self, input_pdf_path: str) -> Dict:
        plumber_pdf_object = pdfplumber.open(input_pdf_path)

        page_to_line_to_tokens = {}
        for page_id in range(len(plumber_pdf_object.pages)):
            cur_page = plumber_pdf_object.pages[page_id]
            page_tokens = self._load_page_tokens(
                page=cur_page,
                page_index=page_id,
                x_tolerance=self.token_x_tolerance,
                y_tolerance=self.token_y_tolerance,
                keep_blank_chars=self.keep_blank_chars,
                use_text_flow=self.use_text_flow,
                horizontal_ltr=self.horizontal_ltr,
                vertical_ttb=self.vertical_ttb,
                extra_attrs=self.extra_attrs,
            )
            line_to_tokens = simple_line_detection(
                page_tokens=page_tokens,
                x_tolerance=self.line_x_tolerance/cur_page.width,
                y_tolerance=self.line_y_tolerance/cur_page.height,
            )
            page_to_line_to_tokens[page_id] = line_to_tokens

        return page_to_line_to_tokens

    def _convert_nested_text_to_doc_json(self, page_to_row_to_tokens: Dict) -> Dict:
        """Copied from sscraper._convert_nested_text_to_doc_json"""
        text = ""
        page_annos: List[SpanGroup] = []
        token_annos: List[SpanGroup] = []
        row_annos: List[SpanGroup] = []
        start = 0
        for page, row_to_tokens in page_to_row_to_tokens.items():
            page_rows: List[SpanGroup] = []
            for row, tokens in row_to_tokens.items():
                # process tokens in this row
                row_tokens: List[SpanGroup] = []
                for k, token in enumerate(tokens):
                    text += token["text"]
                    end = start + len(token["text"])
                    # make token
                    token = SpanGroup(
                        spans=[Span(start=start, end=end, box=token["bbox"])]
                    )
                    row_tokens.append(token)
                    token_annos.append(token)
                    if k < len(tokens) - 1:
                        text += " "
                    else:
                        text += "\n"  # start newline at end of row
                    start = end + 1
                # make row
                row = SpanGroup(
                    spans=[
                        Span(
                            start=row_tokens[0][0].start,
                            end=row_tokens[-1][0].end,
                            box=Box.small_boxes_to_big_box(
                                boxes=[span.box for t in row_tokens for span in t]
                            ),
                        )
                    ]
                )
                page_rows.append(row)
                row_annos.append(row)
            # make page
            if page_rows:
                page = SpanGroup(
                    spans=[
                        Span(
                            start=page_rows[0][0].start,
                            end=page_rows[-1][0].end,
                            box=Box.small_boxes_to_big_box(
                                boxes=[span.box for r in page_rows for span in r]
                            ),
                        )
                    ]
                )
            else:
                page = SpanGroup(spans=[])
            page_annos.append(page)
        # add IDs
        for i, page in enumerate(page_annos):
            page.id = i
        for j, row in enumerate(row_annos):
            row.id = j
        for k, token in enumerate(token_annos):
            token.id = k
        return {
            Symbols: text,
            Pages: [page.to_json() for page in page_annos],
            Tokens: [token.to_json() for token in token_annos],
            Rows: [row.to_json() for row in row_annos],
        }

    def _load_pdf_as_doc(self, input_pdf_path: str) -> Document:

        page_to_line_to_tokens = self._load_pdf_tokens(input_pdf_path)
        doc_json = self._convert_nested_text_to_doc_json(page_to_line_to_tokens)
        doc = Document.from_json(doc_json)
        return doc