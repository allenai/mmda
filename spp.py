#!/usr/bin/env python3
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import requests

from mmda.types.span import Span
from mmda.types.annotation import SpanGroup
from mmda.parsers.pdfplumber_parser import PDFPlumberParser, _SPLIT_AT_PUNCTUATION
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor


def get_elements(sha):
    resp1 = requests.get(f'http://annotations-api.prod.s2.allenai.org/pdf/{sha}/annotations'
                         '?annotationSources=none&attributeSources=pdfplumber-0.0.4').json()
    text_url = resp1['attributesFromSource']['pdfplumber-0.0.4']['text']
    resp2 = requests.post(
        'http://annotations-api.prod.s2.allenai.org/plain-text/',
        json={'s3Url': text_url}
    ).json()
    text_id = resp2['id']
    resp3 = requests.get(f'http://annotations-api.prod.s2.allenai.org/plain-text/{text_id}/annotations'
                         '?annotationSources=vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4'
                         '&attributeSources=none').json()
    annos = resp3['annotationsFromSource']['vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4']['vila_span_groups']
    spangroups = []
    for anno in annos:
        spangroups.append(
            SpanGroup(
                spans=[Span(start=anno['startChar'], end=anno['endChar'])],
                type=anno['attributesFromSource']['vila-0.0.2|layout-parser-0.0.2|pdfplumber-0.0.4']['type']
            )
        )
    return spangroups


def print_title_abstract(doc):
    title_spangroups = [sg for sg in doc.pages[0].elements if sg.type == "Title"]
    abs_spangroups = [sg for sg in doc.elements if sg.type == "Abstract"]

    title_str = " ".join(" ".join(w.text for w in sg.words) for sg in title_spangroups)
    abstract_str = " ".join(" ".join(w.text for w in sg.words) for sg in abs_spangroups)
    print(f'title = "{title_str}"')
    print(f'abstract = "{abstract_str}"')


sha = '17ada3c96ef888fc89f6b995d2edd19a45383423'

doc = PDFPlumberParser(split_at_punctuation=_SPLIT_AT_PUNCTUATION).parse(f"{os.environ['HOME']}/{sha}.pdf")
doc.annotate(elements=get_elements(sha))
doc.annotate(words=DictionaryWordPredictor("/dev/null").predict(doc))
print_title_abstract(doc)
print()

doc = PDFPlumberParser().parse(f"{os.environ['HOME']}/{sha}.pdf")
doc.annotate(elements=get_elements(sha))
doc.annotate(words=DictionaryWordPredictor("/dev/null").predict(doc))
print_title_abstract(doc)
