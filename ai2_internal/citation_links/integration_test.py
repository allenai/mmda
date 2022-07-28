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

from ai2_internal.citation_links.interface import Instance, Prediction
from mmda.types import api


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
        mention1 = api.SpanGroup(
            spans = [],
            box_group = None,
            id = 1,
            type = None,
            text = "4"
        )

        mention2 = api.SpanGroup(
            spans = [],
            box_group = None,
            id = 2,
            type = None,
            text = "16"
        )

        bib1 = api.SpanGroup(
            spans = [],
            box_group = None,
            id = 1,
            type = None,
            text = "[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram, Journal of the American College of Cardiology , vol. 36, no. 6, pp. 1737–1745, 2000."
        )

        bib2 = api.SpanGroup(
            spans = [],
            box_group = None,
            id = 2,
            type = None,
            text = "4. Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding. Computational Intelligence and Neuroscience , 2019. doi: 10.1155/2019/8106073."
        )

        bib3 = api.SpanGroup(
            spans = [],
            box_group = None,
            id = 3,
            type = None,
            text = "37 Urban Taco Collective, Tacos with sauce, 2019"
        )



        instances = [
            Instance(mentions = [mention1, mention2], bibs = [bib1, bib2, bib3])
        ]
        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 1)

        predicted_links = predictions[0].linked_mentions        
        self.assertEqual(len(predicted_links), 2)
        self.assertEqual(predicted_links[0], (mention1, str(bib2.id)))
        self.assertEqual(predicted_links[1], (mention2, str(bib1.id)))
        
