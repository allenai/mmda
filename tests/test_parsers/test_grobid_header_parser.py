import unittest
import unittest.mock as um

import pytest
from mmda.parsers.grobid_parser import GrobidHeaderParser

XML_OK = open("tests/fixtures/grobid-tei-maml-header.xml").read()
XML_NO_TITLE = open("tests/fixtures/grobid-tei-no-title.xml").read()
XML_NO_ABS = open("tests/fixtures/grobid-tei-no-abstract.xml").read()


def mock_post(*args, **kwargs):
    class MockResponse:
        def __init__(self, xml: str, status_code: int) -> None:
            self._xml = xml
            self._status_code = status_code

        @property
        def text(self):
            return self._xml

        @property
        def status_code(self):
            return self._status_code

    if args[0].endswith("ok"):
        return MockResponse(XML_OK, 200)
    elif args[0].endswith("no-title"):
        return MockResponse(XML_NO_TITLE, 200)
    elif args[0].endswith("no-abs"):
        return MockResponse(XML_NO_ABS, 200)
    elif args[0].endswith("err"):
        return MockResponse(None, 500)

    return MockResponse(None, 404)


class TestGrobidHeaderParser(unittest.TestCase):
    @um.patch("requests.post", side_effect=mock_post)
    def test_processes_header(self, mock_post):
        parser = GrobidHeaderParser(url="http://localhost/ok")

        with um.patch("builtins.open", um.mock_open(read_data="it's xml")):
            document = parser.parse(input_pdf_path="some-location")

        assert document.title[0].text.startswith("Model-Agnostic Meta-Learning")
        assert document.abstract[0].text.startswith("We propose an algorithm")

        assert document.title[0].symbols[0:2] == ["Model-Agnostic", "Meta-Learning"]
        assert document.abstract[0].symbols[0:2] == ["We", "propose"]

    @um.patch("requests.post", side_effect=mock_post)
    def test_processes_header_without_title(self, mock_post):
        parser = GrobidHeaderParser(url="http://localhost/no-title")

        with um.patch("builtins.open", um.mock_open(read_data="it's xml")):
            document = parser.parse(input_pdf_path="some-location")

        assert document.title[0].text == ""

        assert document.abstract[0].text.startswith("We propose an algorithm")
        assert document.abstract[0].symbols[0:2] == ["We", "propose"]

    @um.patch("requests.post", side_effect=mock_post)
    def test_processes_header_without_title(self, mock_post):
        parser = GrobidHeaderParser(url="http://localhost/no-abs")

        with um.patch("builtins.open", um.mock_open(read_data="it's xml")):
            document = parser.parse(input_pdf_path="some-location")

        assert document.abstract[0].text == ""

        assert document.title[0].text.startswith("Model-Agnostic Meta-Learning")
        assert document.title[0].symbols[0:2] == ["Model-Agnostic", "Meta-Learning"]

    @um.patch("requests.post", side_effect=mock_post)
    def test_processes_header_server_error_raises(self, mock_post):
        parser = GrobidHeaderParser(url="http://localhost/err")

        with pytest.raises(RuntimeError) as ex:
            with um.patch("builtins.open", um.mock_open(read_data="it's xml")):
                parser.parse(input_pdf_path="some-location")

        assert "Unable to process" in str(ex.value)
