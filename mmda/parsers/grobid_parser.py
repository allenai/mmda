import io
import xml.etree.ElementTree as et

import requests
from mmda.parsers.parser import Parser
from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.names import Symbols
from mmda.types.span import Span

DEFAULT_API = "http://localhost:8070/api/processHeaderDocument"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _get_token_spans(text: str, tokens: list[str], offset: int = 0) -> list[int]:
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

    def parse(self, input_pdf_path: str) -> Document:
        xml = _post_document(self.url, input_pdf_path)
        root = et.parse(io.StringIO(xml)).getroot()

        title = self._get_title(root)
        if len(title.text) == 0:
            raise RuntimeError(f"Unable to extract title: {input_pdf_path}!")

        abstract = self._get_abstract(root, offset=len(title.text) + 1)
        if len(abstract.text) == 0:
            raise RuntimeError(f"Unable to extract abstract: {input_pdf_path}!")

        symbols = "\n".join([title.text, abstract.text])

        document = Document(symbols=symbols)
        document.annotate(title=[title], abstract=[abstract])

        return document

    def _get_title(self, root: et.Element) -> SpanGroup:
        matches = root.findall(".//tei:titleStmt/tei:title", NS)

        if len(matches) == 0:
            return []

        text = matches[0].text.strip()
        tokens = text.split()
        spans = _get_token_spans(text, tokens)

        return SpanGroup(spans=spans, text=text)

    def _get_abstract(self, root: et.Element, offset: int) -> SpanGroup:
        matches = root.findall(".//tei:profileDesc//tei:abstract//", NS)

        if len(matches) == 0:
            return []

        # An abstract may have many paragraphs
        text = "\n".join(m.text for m in matches)
        tokens = text.split()
        spans = _get_token_spans(text, tokens, offset=offset)

        return SpanGroup(spans=spans, text=text)
