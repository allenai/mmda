import os
import json
import re
import sys

import requests

from mmda.types.annotation import SpanGroup, BoxGroup
from mmda.types.document import Document
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor


assert re.match(".+\\.pdf$", sys.argv[1].lstrip())
pdf_file_prefix = '.'.join(os.path.abspath(sys.argv[1]).split(".")[:-1])

pdf_file = f"{pdf_file_prefix}.pdf"
document_file = f"{pdf_file_prefix}-document.json"
elements_file = f"{pdf_file_prefix}-elements.json"

print(f"reading pdf from from {pdf_file}")
print(f"writing document to   {document_file}")
print(f"writing elements to   {elements_file}")

with open(f"{pdf_file_prefix}.pdf", "rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://localhost:8081", data=pdf_bytes).json())

resp = requests.post("http://localhost:8082", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)

resp = requests.post("http://localhost:8083", json=doc.to_json(with_images=True))
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(vila_spans=span_groups)

dictionary_word_predictor = DictionaryWordPredictor("/dev/null")
words = dictionary_word_predictor.predict(doc)
doc.annotate(words=words)

with open(document_file, "w") as f:
    json.dump(doc.to_json(), f)  # add with_images=True if you want.

title = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == "Title"
)
abstract = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == "Abstract"
)

with open(elements_file, 'w') as f:
    json.dump({'title': title, 'abstract': abstract}, f, indent=2)
