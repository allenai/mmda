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

from ai2_internal import api
from ai2_internal.citation_links.interface import Instance

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)


# single text string representing text contents of pdf
SYMBOLS = "titlexxxx4xxxx16xxx[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram4. Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding.37 Urban Taco Collective, Tacos with sauce, 2019"

SPAN_1 = api.Span(start = 9, end = 10, box = None)
MENTION_1 = api.SpanGroup(
    spans = [SPAN_1],
    box_group = None,
    id = 1,
    type = None,
    text = None
)

# text for this span is "16"
SPAN_2 = api.Span(start = 14, end = 16, box = None)
MENTION_2 = api.SpanGroup(
    spans = [SPAN_2],
    box_group = None,
    id = 2,
    type = None,
    text = None
)

# text for this span is "[16] C. Fisch, Centennial of the string galvanometer and the electro- cardiogram"
BIB_SPAN_1 = api.Span(start = 19, end = 98, box = None)
BIB_1 = api.SpanGroup(
    spans = [BIB_SPAN_1],
    box_group = None,
    id = 1,
    type = None,
    text = None
)

# text for this span is "4. Wei Zhuo, Qianyi Zhan, Yuan Liu, Zhenping Xie, and Jing Lu. Context attention heterogeneous network embed- ding."
BIB_SPAN_2 = api.Span(start = 99, end = 213, box = None)
BIB_2 = api.SpanGroup(
    spans = [BIB_SPAN_2],
    box_group = None,
    id = 2,
    type = None,
    text = None
)

# text for this span is "37 Urban Taco Collective, Tacos with sauce, 2019"
BIB_SPAN_3 = api.Span(start = 214, end = 261, box = None)
BIB_3 = api.SpanGroup(
    spans = [BIB_SPAN_3],
    box_group = None,
    id = 3,
    type = None,
    text = None
)


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [
            Instance(symbols = SYMBOLS, mentions = [MENTION_1, MENTION_2], bibs = [BIB_1, BIB_2, BIB_3])
        ]

        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 1)

        predicted_links = predictions[0].linked_mentions
        self.assertEqual(len(predicted_links), 2)
        self.assertEqual(predicted_links[0], (str(MENTION_1.id), str(BIB_2.id)))
        self.assertEqual(predicted_links[1], (str(MENTION_2.id), str(BIB_1.id)))

        predicted_relations = predictions[0].linked_mention_relations
        self.assertEqual(len(predicted_relations), 2)
        self.assertEqual(predicted_relations[0], api.Relation(from_id=MENTION_1.id, to_id=BIB_2.id))
        self.assertEqual(predicted_relations[1], api.Relation(from_id=MENTION_2.id, to_id=BIB_1.id))
    def test__predictions_predicts_nothing_when_there_are_no_mentions(self, container):
        instances = [
            Instance(symbols = SYMBOLS, mentions = [], bibs = [BIB_1, BIB_2, BIB_3])
        ]

        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 1)

        predicted_relations = predictions[0].linked_mention_relations
        self.assertEqual(len(predicted_relations), 0)

    def test__predictions_predicts_nothing_when_there_are_no_bib_entries(self, container):
        instances = [
            Instance(symbols = SYMBOLS, mentions = [MENTION_1, MENTION_2], bibs = [])
        ]

        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 1)

        predicted_relations = predictions[0].linked_mention_relations
        self.assertEqual(len(predicted_relations), 0)

