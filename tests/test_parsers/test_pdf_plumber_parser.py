"""

@kylel
"""

import os
import pathlib
import unittest

from mmda.types import Document
from mmda.parsers import PDFPlumberParser

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
        assert '[1-5]' in no_split_tokens_with_numbers
        assert 'GPT-3[10]' in no_split_tokens_with_numbers

        custom_split_parser = PDFPlumberParser(split_at_punctuation=',.[]:')
        custom_split_doc = custom_split_parser.parse(input_pdf_path="../fixtures/2107.07170.pdf")
        custom_split_tokens_with_numbers = [token.text for token in custom_split_doc.tokens if re.search(r'[0-9]', token.text)]
        assert '[1-5]' not in custom_split_tokens_with_numbers
        assert '1-5' in custom_split_tokens_with_numbers
        assert 'GPT-3[10]' not in custom_split_tokens_with_numbers
        assert 'GPT-3' in custom_split_tokens_with_numbers

        default_split_parser = PDFPlumberParser(split_at_punctuation=True)
        default_split_doc = default_split_parser.parse(input_pdf_path="../fixtures/2107.07170.pdf")
        default_split_tokens_with_numbers = [token.text for token in default_split_doc.tokens if re.search(r'[0-9]', token.text)]
        assert '1-5' not in default_split_tokens_with_numbers
        assert 'GPT-3' not in default_split_tokens_with_numbers

        assert len(no_split_tokens_with_numbers) < len(custom_split_tokens_with_numbers) < len(default_split_tokens_with_numbers)


    def test_align_coarse_and_fine_tokens(self):
        parser = PDFPlumberParser()

        # example
        coarse_tokens = ['abc', 'def']
        fine_tokens = ['ab', 'c', 'd', 'ef']
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens,
            fine_tokens=fine_tokens
        )
        assert out == [0, 0, 1, 1]

        # minimal case
        coarse_tokens = []
        fine_tokens = []
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens,
            fine_tokens=fine_tokens
        )
        assert out == []

        # identical case
        coarse_tokens = ['a', 'b', 'c']
        fine_tokens = ['a', 'b', 'c']
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens,
            fine_tokens=fine_tokens
        )
        assert out == [0, 1, 2]

        # misaligned case
        with self.assertRaises(AssertionError):
            coarse_tokens = ['a', 'b']
            fine_tokens = ['ab']
            parser._align_coarse_and_fine_tokens(
                coarse_tokens=coarse_tokens,
                fine_tokens=fine_tokens
            )

        # same num of chars, but chars mismatch case
        with self.assertRaises(AssertionError):
            coarse_tokens = ['ab']
            fine_tokens = ['a', 'c']
            parser._align_coarse_and_fine_tokens(
                coarse_tokens=coarse_tokens,
                fine_tokens=fine_tokens
            )
