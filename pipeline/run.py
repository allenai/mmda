import sys
import requests
import csv
import json
from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.document import Document

pdf_file = sys.argv[1]

with open(pdf_file,"rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://symbolscraper", data = pdf_bytes).json())

resp = requests.post("http://layoutparser", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)

resp = requests.post("http://vila", json=doc.to_json(with_images=True))
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(preds=span_groups)

with open("/pipeline/output.json","w") as f:
    json.dump(doc.to_json(), f)

with open(f"/pipeline/output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows((s.type, "\n".join(s.symbols)) for s in doc.preds)



