import os
import pathlib
import unittest

from mmda.types.document import Document
from mmda.parsers.pdfplumber_parser import PDFPlumberParser

import re

os.chdir(pathlib.Path(__file__).parent)


class TestPDFPlumberParser(unittest.TestCase):
    
    def test_parse(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path="../fixtures/1903.10676.pdf")
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

    def test_split_punctuation(self):
        no_split_parser = PDFPlumberParser(split_at_punctuation=False)
        no_split_doc = no_split_parser.parse(input_pdf_path="../fixtures/2107.07170.pdf")
        no_split_tokens_with_numbers = [token.text for token in no_split_doc.tokens if re.search(r'[0-9]', token.text)]

        split_parser = PDFPlumberParser(split_at_punctuation=True)
        split_doc = split_parser.parse(input_pdf_path="../fixtures/2107.07170.pdf")
        split_tokens_with_numbers = [token.text for token in split_doc.tokens if re.search(r'[0-9]', token.text)]

        custom_split_parser = PDFPlumberParser(split_at_punctuation='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~–§')
        custom_split_doc = custom_split_parser.parse(input_pdf_path="../fixtures/2107.07170.pdf")
        custom_split_tokens_with_numbers = [token.text for token in custom_split_doc.tokens if re.search(r'[0-9]', token.text)]

        assert len(no_split_tokens_with_numbers) < len(split_tokens_with_numbers) < len(custom_split_tokens_with_numbers)

