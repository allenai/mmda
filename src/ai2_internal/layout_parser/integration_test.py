"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.
"""


import logging
import sys
import unittest

from .interface import Instance

from PIL import Image
from pathlib import Path

import os.path

from mmda.types.image import tobase64

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)

@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):

    def get_images(self):
        return [
            tobase64(
                Image.open(os.path.join(Path(os.path.dirname(__file__)).parent, "shared_test_fixtures", "page0.png"))
            )
        ]

    def test__predictions(self, container):

        instances = [Instance(page_images=self.get_images())]
        predictions = container.predict_batch(instances)

        for prediction in predictions:
            self.assertEqual(
                set(g.type for g in prediction.groups),
                {"Text", "Title"},
            )
