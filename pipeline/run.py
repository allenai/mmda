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
parser_file = f"{pdf_file_prefix}-parser.json"
blocks_file = f"{pdf_file_prefix}-blocks.json"
spans_file = f"{pdf_file_prefix}-spans.json"
words_file = f"{pdf_file_prefix}-words.json"

print(f"reading pdf from from {pdf_file}")
print(f"writing document to   {document_file}")
print(f"writing elements to   {elements_file}")
print(f"writing parser   to   {parser_file}")
print(f"writing blocks   to   {blocks_file}")
print(f"writing spans    to   {spans_file}")
print(f"writing words    to   {words_file}")

with open(f"{pdf_file_prefix}.pdf", "rb") as f:
    pdf_bytes = f.read()

doc = Document.from_json(requests.post("http://localhost:8081", data=pdf_bytes).json())
with open(parser_file, "w") as f:
    json.dump(doc.to_json(), f, indent=2)

resp = requests.post("http://localhost:8082", json=doc.to_json(with_images=True))
box_groups = [BoxGroup.from_json(p) for p in resp.json()]
doc.annotate(blocks=box_groups)
with open(blocks_file, "w") as f:
    json.dump([bg.to_json() for bg in box_groups], f, indent=2)

resp = requests.post("http://localhost:8083", json=doc.to_json(with_images=True))
span_groups = [SpanGroup.from_json(p) for p in resp.json()]
doc.annotate(vila_spans=span_groups)
with open(spans_file, "w") as f:
    json.dump([sg.to_json() for sg in span_groups], f, indent=2)

dictionary_word_predictor = DictionaryWordPredictor("/dev/null")
words = dictionary_word_predictor.predict(doc)
doc.annotate(words=words)
with open(words_file, "w") as f:
    json.dump([sg.to_json() for sg in words], f, indent=2)

with open(document_file, "w") as f:
    json.dump(doc.to_json(), f)  # add with_images=True if you want.

title = ' '.join([' '.join(sg.symbols) for sg in doc.vila_spans if sg.type == "Title"])
abstract = ' '.join([' '.join(sg.symbols) for sg in doc.vila_spans if sg.type == "Abstract"])
abstract_cleaned = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == "Abstract"
)

with open(elements_file, 'w') as f:
    json.dump({'title': title, 'abstract': abstract, 'abstract_cleaned': abstract_cleaned}, f, indent=2)
