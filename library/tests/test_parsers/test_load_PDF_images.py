"""

Tests for PDF Image parser

@kylel, @shannons


"""

import unittest
import tempfile
import json
import os

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
        image_str = images[0].tobase64()

        assert hasattr(images[0], "tobase64")

        recovered_image = Image.frombase64(image_str)
        diff = ImageChops.difference(recovered_image, images[0])

        assert diff.getbbox() is None

        with tempfile.TemporaryDirectory() as outdir:
            with open(os.path.join(outdir, "image.json"), "w") as fp:
                json.dump({"image": [image_str]}, fp)
