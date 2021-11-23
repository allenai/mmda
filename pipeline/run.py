import os
import json
import sys

import requests

from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.document import Document

pdf_file = os.path.abspath(sys.argv[1])
json_file = f"{pdf_file}.json"

with open(pdf_file, "rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://symbolscraper", data=pdf_bytes).json())

resp = requests.post("http://layoutparser", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)

resp = requests.post("http://vila", json=doc.to_json(with_images=True))
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(preds=span_groups)

with open(json_file, "w") as f:
    json.dump(doc.to_json(), f)
