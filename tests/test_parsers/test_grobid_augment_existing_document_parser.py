import json
import logging
import os
import pathlib
import unittest
import unittest.mock as um

import pytest
from mmda.types.document import Document
from mmda.parsers.grobid_augment_existing_document_parser import GrobidAugmentExistingDocumentParser

os.chdir(pathlib.Path(__file__).parent)

PDF_PATH = "../fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4.pdf"
PDFPLUMBER_DOC_PATH = "../fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4__pdfplumber_doc.json"

OK_CONFIG_PATH =  "../fixtures/grobid_augment_existing_document_parser/grobid.config"
XML_OK = open("../fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4.xml").read()

NO_AUTHORS_CONFIG_PATH =  "../fixtures/grobid_augment_existing_document_parser/grobid-no-authors.config"
XML_NO_AUTHORS = open("../fixtures/grobid_augment_existing_document_parser/e5910c027af0ee9c1901c57f6579d903aedee7f4_no_authors.xml").read()

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

    # config file url is used to determine which XML to return from mock Grobid server
    if args[1].startswith("ok"):
        return MockResponse(XML_OK, 200)
    elif args[1].startswith("no-authors"):
        return MockResponse(XML_NO_AUTHORS, 200)

    return MockResponse(None, 404)


class TestGrobidAugmentExistingDocumentParser(unittest.TestCase):
    @um.patch("requests.request", side_effect=mock_request)
    def test_processes_full_text(self, mock_request):
        with open(PDFPLUMBER_DOC_PATH) as f_in:
            doc_dict = json.load(f_in)
            doc = Document.from_json(doc_dict)

        augmenter_parser = GrobidAugmentExistingDocumentParser(config_path=OK_CONFIG_PATH, check_server=False)

        augmented_doc = augmenter_parser.parse(input_pdf_path=PDF_PATH, doc=doc)

        assert len(augmented_doc.bib_entries) is 40
        assert augmented_doc.bib_entries[0].text.startswith("ISPRS 2D Semantic Labeling Challenge.")
        assert len(augmented_doc.authors) is 4
        for m in augmented_doc.citation_mentions:
            print(m.text)
        assert len(augmented_doc.citation_mentions) is 67
        assert 1 == 0

    @um.patch("requests.request", side_effect=mock_request)
    def test_passes_if_xml_missing_authors(self, mock_request):
        with open(PDFPLUMBER_DOC_PATH) as f_in:
            doc_dict = json.load(f_in)
            doc = Document.from_json(doc_dict)

        augmenter_parser = GrobidAugmentExistingDocumentParser(config_path=NO_AUTHORS_CONFIG_PATH, check_server=False)

        augmented_doc = augmenter_parser.parse(input_pdf_path=PDF_PATH, doc=doc)

        assert len(augmented_doc.authors) is 0
