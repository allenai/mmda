#!/usr/bin/env python3
import boto3
import requests

from mmda.types.span import Span
from mmda.types.annotation import SpanGroup
from mmda.types.document import Document


def get_text(url: str) -> str:
    s3 = boto3.resource('s3')
    url_no_prefix = url[5:]
    bucket, *key = url_no_prefix.split("/")
    key = "/".join(key)
    with s3.Bucket(bucket).Object(key).get()['Body'] as f:
        return f.read().decode('utf-8')

sha = '17ada3c96ef888fc89f6b995d2edd19a45383423'
resp1 = requests.get(f'http://annotations-api.prod.s2.allenai.org/pdf/{sha}/annotations'
                     '?annotationSources=none&attributeSources=pdfplumber-0.0.4').json()
text_url = resp1['attributesFromSource']['pdfplumber-0.0.4']['text']
print(text_url)

resp2 = requests.post(
    'http://annotations-api.prod.s2.allenai.org/plain-text/',
    json={'s3Url': text_url}
).json()
text_id = resp2['id']
print(text_id)

resp3 = requests.get(f'http://annotations-api.prod.s2.allenai.org/plain-text/{text_id}/annotations'
                     '?annotationSources=vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4'
                     '&attributeSources=none').json()
annos = resp3['annotationsFromSource']['vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4']['vila_span_groups']

text = get_text(text_url)
spangroups = []
for anno in annos:
    spangroups.append(
        SpanGroup(
            spans=[Span(start=anno['startChar'], end=anno['endChar'])],
            type=anno['attributesFromSource']['vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4']['type']
        )
    )

doc = Document(text)
doc.annotate(vila=spangroups)

for spangroup in doc.vila:
    if spangroup.type == "Bibliography":
        print(spangroup.symbols)
