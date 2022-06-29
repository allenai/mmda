import os
import pathlib
import unittest

from mmda.types.document import Document
from mmda.parsers.pdfplumber_parser import PDFPlumberParser

os.chdir(pathlib.Path(__file__).parent)

PDFFILEPATH = "../fixtures/1903.10676.pdf"

class TestPDFPlumberParser(unittest.TestCase):
    
    def test_parse(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=PDFFILEPATH)
        # right output type
        assert isinstance(doc, Document)
        # the right fields
        assert doc.symbols
        assert doc.pages
        assert doc.tokens
        assert doc.rows
        # roughly the right content
        for keyword in ['Field', 'Task', 'SOTA', 'Base', 'Frozen', 'Finetune', 'NER']:
            assert keyword in doc.symbols[:100]
        