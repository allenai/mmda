"""

@rauthur, @kylel

"""

import os
import io
import xml.etree.ElementTree as et
from typing import List, Optional, Text
import requests
import tempfile
import json

from mmda.parsers.parser import Parser
from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.metadata import Metadata
from mmda.types.span import Span

DEFAULT_API = "http://localhost:8070/api/processHeaderDocument"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _null_span_group() -> SpanGroup:
    sg = SpanGroup(spans=[])
    return sg


def _get_token_spans(text: str, tokens: List[str], offset: int = 0) -> List[int]:
    assert len(text) > 0
    assert len(tokens) > 0
    assert offset >= 0

    spans = [Span(start=offset, end=len(tokens[0]) + offset)]

    for i, token in enumerate(tokens):
        if i == 0:
            continue

        start = text.find(token, spans[-1].end - offset, len(text))
        end = start + len(token)

        spans.append(Span(start=start + offset, end=end + offset))

    return spans


def _post_document(url: str, input_pdf_path: str) -> str:
    req = requests.post(url, files={"input": open(input_pdf_path, "rb")})

    if req.status_code != 200:
        raise RuntimeError(f"Unable to process document: {input_pdf_path}!")

    return req.text

class GrobidHeaderParser(Parser):
    """Grobid parser that uses header API methods to get title and abstract only. The
    current purpose of this class is evaluation against other methods for title and
    abstract extraction from a PDF.
    """

    _url: str

    def __init__(self, url: str = DEFAULT_API) -> None:
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    def parse(self, input_pdf_path: str, tempdir: Optional[str] = None) -> Document:
        xml = _post_document(url=self.url, input_pdf_path=input_pdf_path)

        if tempdir:
            os.makedirs(tempdir, exist_ok=True)
            xmlfile = os.path.join(tempdir, os.path.basename(input_pdf_path).replace('.pdf', '.xml'))
            with open(xmlfile, 'w') as f_out:
                f_out.write(xml)

        doc: Document = self._parse_xml_to_doc(xml=xml)
        return doc

    def _parse_xml_to_doc(self, xml: str) -> Document:
        root = et.parse(io.StringIO(xml)).getroot()

        title = self._get_title(root=root)

        # Here we +1 len because we add a "\n" later when joining (if title found)
        abstract_offset = 0 if len(title.text) == 0 else (len(title.text) + 1)
        abstract = self._get_abstract(root=root, offset=abstract_offset)

        symbols = "\n".join([t for t in [title.text, abstract.text] if len(t) > 0])

        document = Document(symbols=symbols)
        document.annotate(title=[title], abstract=[abstract])

        return document

    def _get_title(self, root: et.Element) -> SpanGroup:
        matches = root.findall(".//tei:titleStmt/tei:title", NS)

        if len(matches) == 0:
            return _null_span_group()

        if not matches[0].text:
            return _null_span_group()

        text = matches[0].text.strip()
        tokens = text.split()
        spans = _get_token_spans(text, tokens)

        sg = SpanGroup(spans=spans, metadata=Metadata(text=text))
        return sg

    def _get_abstract(self, root: et.Element, offset: int) -> SpanGroup:
        matches = root.findall(".//tei:profileDesc//tei:abstract//", NS)

        if len(matches) == 0:
            return _null_span_group()

        # An abstract may have many paragraphs
        text = "\n".join(m.text for m in matches)
        tokens = text.split()
        spans = _get_token_spans(text, tokens, offset=offset)

        sg = SpanGroup(spans=spans, metadata=Metadata(text=text))
        return sg
