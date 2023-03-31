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

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest

from .interface import Instance, Prediction


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
    def test__predictions(self, container):
        instances = [
            Instance(bib_entry="[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram, Journal of the American College of Cardiology , vol. 36, no. 6, pp. 1737–1745, 2000."),
            Instance(bib_entry="Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding. Computational Intelligence and Neuroscience , 2019. doi: 10.1155/2019/8106073."),
        ]
        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 2)

        expected1 = Prediction(
            citation_number="16",
            title="Centennial of the string galvanometer and the electrocardiogram",
            doi=None
        )
        self.assertEqual(predictions[0], expected1)

        expected2 = Prediction(
            citation_number=None,
            title="Context attention heterogeneous network embedding",
            doi="10.1155/2019/8106073"
        )
        self.assertEqual(predictions[1], expected2)
