import logging
import os
import pathlib
import unittest
import unittest.mock as um

import pytest
from mmda.parsers.grobid_augment_existing_document_parser import GrobidAugmentExistingDocumentParser
from mmda.parsers.pdfplumber_parser import PDFPlumberParser

os.chdir(pathlib.Path(__file__).parent)

PDF_PATH = "../fixtures/grobid_augment_existing_document_parser/0c027af0ee9c1901c57f6579d903aedee7f4.pdf"
XML_OK = open("../fixtures/grobid_augment_existing_document_parser/0c027af0ee9c1901c57f6579d903aedee7f4.xml").read()
TEST_CONFIG_PATH =  "../fixtures/grobid_augment_existing_document_parser/grobid.config"

def mock_request(*args, **kwargs):
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

    return MockResponse(XML_OK, 200)


class TestGrobidAugmentExistingDocumentParser(unittest.TestCase):
    @um.patch("requests.request", side_effect=mock_request)
    def test_processes_full_text(self, mock_request):
        logging.getLogger("pdfminer").setLevel(logging.WARNING)

        pdf_plumber = PDFPlumberParser()
        doc = pdf_plumber.parse(input_pdf_path=PDF_PATH)
        print("mmmkay??")

        config_data = open(TEST_CONFIG_PATH).read()
        with um.patch("builtins.open", um.mock_open(read_data=config_data)):
            parser = GrobidAugmentExistingDocumentParser(config_path=TEST_CONFIG_PATH, check_server=False)
        print("parser initited")

        doc = parser.parse(input_pdf_path=PDF_PATH, doc=doc)
        
        assert len(doc.bib_entries) is 40
        assert doc.bib_entries[0].text.startswith("ISPRS 2D Semantic Labeling Challenge.")
