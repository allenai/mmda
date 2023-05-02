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

import json
import logging
import os
import sys
import unittest

from .. import api
from mmda.types.document import Document
from .interface import Instance
import mmda.types as mmda_types

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)

def resolve(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'test_fixtures', file)

def get_test_instance(sha) -> Instance:
    doc_file = resolve(f'test_doc_sha_{sha}.json')
    with open(doc_file) as f:
        dic_json = json.load(f)
    doc = Document.from_json(dic_json['doc'])
    layout_equations = [mmda_types.BoxGroup.from_json(entry) for entry in dic_json['layout_equations']]
    tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
    vila_span_groups = [api.SpanGroup.from_mmda(sg) for sg in doc.vila_span_groups]
    blocks = [api.BoxGroup.from_mmda(bg) for bg in layout_equations]

    return Instance(
        symbols=doc.symbols,
        tokens=tokens,
        pages=pages,
        vila_span_groups=vila_span_groups,
        blocks=blocks,
    )


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions_test_doc_sha_d045(self, container):
        instances = [get_test_instance('d0450478c38dda61f9943f417ab9fcdb2ebeae0a')]
        predictions = container.predict_batch(instances)
        assert isinstance(predictions[0].figures[0], api.BoxGroup)
        assert isinstance(predictions[0].figure_captions[0], api.SpanGroup)
        assert isinstance(predictions[0].figure_to_figure_captions[0], api.Relation)
        assert isinstance(predictions[0].tables[0], api.BoxGroup)
        assert isinstance(predictions[0].table_captions[0], api.SpanGroup)
        assert isinstance(predictions[0].table_to_table_captions[0], api.Relation)
        assert len(predictions[0].figures) == 5
        assert len(predictions[0].tables) == 6

    def test__predictions_test_doc_sha_08f05(self, container):
        instances = [get_test_instance('08f02e7888f140a76a00ed23fce2f2fc303a')]
        predictions = container.predict_batch(instances)
        assert isinstance(predictions[0].figures[0], api.BoxGroup)
        assert isinstance(predictions[0].figure_captions[0], api.SpanGroup)
        assert isinstance(predictions[0].figure_to_figure_captions[0], api.Relation)
        assert isinstance(predictions[0].tables[0], api.BoxGroup)
        assert isinstance(predictions[0].table_captions[0], api.SpanGroup)
        assert isinstance(predictions[0].table_to_table_captions[0], api.Relation)
        assert len(predictions[0].figures) == 4
        assert len(predictions[0].tables) == 3
