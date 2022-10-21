import os
import pathlib
import unittest

from mmda.types import Document
from mmda.parsers import PDFPlumberParser

import re


class TestPDFPlumberParser(unittest.TestCase):
    def setUp(cls) -> None:
        cls.fixture_path = os.path.join(pathlib.Path(__file__).parent.parent, 'fixtures')

    def test_parse(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=self.fixture_path / '1903.10676.pdf')
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
        no_split_doc = no_split_parser.parse(input_pdf_path=self.fixture_path / '2107.07170.pdf')
        no_split_tokens_with_numbers = [token.text for token in no_split_doc.tokens if re.search(r'[0-9]', token.text)]
        assert '[1-5]' in no_split_tokens_with_numbers
        assert 'GPT-3[10]' in no_split_tokens_with_numbers

        custom_split_parser = PDFPlumberParser(split_at_punctuation=',.[]:')
        custom_split_doc = custom_split_parser.parse(
            input_pdf_path=self.fixture_path / '2107.07170.pdf'
        )
        custom_split_tokens_with_numbers = [token.text for token in custom_split_doc.tokens if re.search(r'[0-9]',
                                                                                                         token.text)]
        assert '[1-5]' not in custom_split_tokens_with_numbers
        assert '1-5' in custom_split_tokens_with_numbers
        assert 'GPT-3[10]' not in custom_split_tokens_with_numbers
        assert 'GPT-3' in custom_split_tokens_with_numbers

        default_split_parser = PDFPlumberParser(split_at_punctuation=True)
        default_split_doc = default_split_parser.parse(input_pdf_path=os.path.join(self.fixture_path,
                                                                                   '2107.07170.pdf'))
        default_split_tokens_with_numbers = [token.text for token in default_split_doc.tokens if re.search(r'[0-9]',
                                                                                                           token.text)]
        assert '1-5' not in default_split_tokens_with_numbers
        assert 'GPT-3' not in default_split_tokens_with_numbers

        assert len(no_split_tokens_with_numbers) < len(custom_split_tokens_with_numbers) < len(default_split_tokens_with_numbers)

