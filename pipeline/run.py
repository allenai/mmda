import os
import json
import sys

import requests

from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.document import Document

pdf_file = os.path.abspath(sys.argv[1])
json_file = f"{pdf_file}.json"
json2_file = f"{pdf_file}.json2"

with open(pdf_file, "rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://localhost:8081", data=pdf_bytes).json())

resp = requests.post("http://localhost:8082", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)

resp = requests.post("http://localhost:8083", json=doc.to_json(with_images=True))
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(preds=span_groups)

with open(json_file, "w") as f:
    json.dump(doc.to_json(), f)

result = {
    'abstract': [],
    'title': []
}

for span_group in doc.preds:
    if span_group.type == 'abstract':
        result['abstract'].append(span_group)
    if span_group.type == 'title':
        result['title'].append(span_group)

title = ' '.join([' '.join(x.symbols) for x in result['title']])
abstract = '\n'.join([' '.join(x.symbols) for x in result['abstract']])

with open(json2_file, 'w') as f:
    json.dump({'title': title, 'abstract': abstract}, f, indent=2)
