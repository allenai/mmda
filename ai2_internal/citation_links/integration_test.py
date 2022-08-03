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
        # single text string representing text contents of pdf
        symbols = "titlexxxx4xxxx16xxx[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram4. Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding.37 Urban Taco Collective, Tacos with sauce, 2019"
        
        # text for this span is "4"
        span1 = api.Span(start = 9, end = 10, box = None)
        mention1 = api.SpanGroup(
            spans = [span1],
            box_group = None,
            id = 1,
            type = None,
            text = None
        )

        # text for this span is "16"
        span2 = api.Span(start = 14, end = 16, box = None)
        mention2 = api.SpanGroup(
            spans = [span2],
            box_group = None,
            id = 2,
            type = None,
            text = None
        )

        # text for this span is "[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram
        bibspan1 = api.Span(start = 19, end = 98, box = None)
        bib1 = api.SpanGroup(
            spans = [bibspan1],
            box_group = None,
            id = 1,
            type = None,
            text = None
        )

        # text for this span is "4. Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding."
        bibspan2 = api.Span(start = 99, end = 213, box = None)
        bib2 = api.SpanGroup(
            spans = [bibspan2],
            box_group = None,
            id = 2,
            type = None,
            text = None
        )

        # text for this span is "37 Urban Taco Collective, Tacos with sauce, 2019"
        bibspan3 = api.Span(start = 214, end = 261, box = None)
        bib3 = api.SpanGroup(
            spans = [bibspan3],
            box_group = None,
            id = 3,
            type = None,
            text = None
        )


        instances = [
            Instance(symbols = symbols, mentions = [mention1, mention2], bibs = [bib1, bib2])
        ]

        
        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 1)

        predicted_links = predictions[0].linked_mentions        
        self.assertEqual(len(predicted_links), 2)
        self.assertEqual(predicted_links[0], (mention1, str(bib2.id)))
        self.assertEqual(predicted_links[1], (mention2, str(bib1.id)))
        
