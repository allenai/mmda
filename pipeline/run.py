import os
import json
import sys
from uuid import UUID

import requests

from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.document import Document
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        return json.JSONEncoder.default(self, obj)


pdf_file = os.path.abspath(sys.argv[1])
json_file = f"{pdf_file}.json"
json2_file = f"{pdf_file}.json2"

with open(pdf_file, "rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://localhost:8081", data=pdf_bytes).json())

resp = requests.post("http://localhost:8082", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)

resp = requests.post(
    "http://localhost:8083",
    data=json.dumps(doc.to_json(with_images=True), cls=UUIDEncoder),
    headers={"Content-Type": "application/json"}
)
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(vila_spans=span_groups)

dictionary_word_predictor = DictionaryWordPredictor("/dev/null")
words = dictionary_word_predictor.predict(doc)
doc.annotate(words=words)

with open(json_file, "w") as f:
    json.dump(doc.to_json(), f, cls=UUIDEncoder)  # add with_images=True if you want.

title = ' '.join([' '.join(sg.symbols) for sg in doc.vila_spans if sg.type == "Title"])
abstract = '\n'.join([' '.join(sg.symbols) for sg in doc.vila_spans if sg.type == "Abstract"])
abstract_cleaned = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == "Abstract"
)

with open(json2_file, 'w') as f:
    json.dump({'title': title, 'abstract': abstract, 'abstract_cleaned': abstract_cleaned}, f, indent=2)
