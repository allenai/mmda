import os
import requests
from unittest import TestCase
import json

from mmda.types.document import Document
from mmda.types.annotation import SpanGroup
from mmda.types.image import Image


URL = "http://localhost:8080/invocations"


def get_test_document():
    data_file = os.path.join(os.path.dirname(__file__), "data", "test_doc.json")
    with open(data_file, "r") as f:
        doc = json.load(f)
    doc["images"] = [
        Image.open(
            os.path.join(os.path.dirname(__file__), "data", "page0.png")
        ).tobase64()
    ]

    return Document.from_json(doc).to_json(with_images=True)


class TestInvocations(TestCase):
    def test__live_invocation(self):
        request = {"instances": [get_test_document()]}

        resp = requests.post(URL, json=request)

        self.assertEqual(resp.status_code, 200)
        for pred in resp.json()["predictions"]:
            annotations = [SpanGroup.from_json(sg) for sg in pred]
            annotation_types = set(a.type for a in annotations)
            self.assertEqual({"Bibliography"}, annotation_types)

    def test_jsonl_invocation(self):
        doc = get_test_document()
        docs = [doc, doc]
        request = "\n".join([json.dumps(d) for d in docs])

        resp = requests.post(
            URL,
            data=request,
            headers={
                "content-type": "application/jsonlines",
                "accept": "application/jsonlines",
            },
        )
        self.assertEqual(resp.status_code, 200)
        for line in resp.text.split("\n"):
            pred = json.loads(line)
            annotations = [SpanGroup.from_json(sg) for sg in pred]
            annotation_types = set(a.type for a in annotations)
            self.assertEqual({"Bibliography"}, annotation_types)
