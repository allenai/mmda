"""

Tests for PDF Image parser

@kylel


"""

import unittest

from PIL import ImageChops

from mmda.parsers.parser import BaseParser
from mmda.types.image import Image

class TestLoadPDFImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_path = "tests/fixtures/1903.10676.pdf"
        cls.parser = BaseParser()

    def test_load_image(self):
        images = self.parser.load_images(self.pdf_path)

        assert hasattr(images[0], "tobase64")

        recovered_image = Image.frombase64(images[0].tobase64()) 
        diff = ImageChops.difference(recovered_image, images[0])

        assert diff.getbbox() is None