import subprocess
import requests
import json
from unittest import TestCase
from mmda.types.image import Image
from mmda.types.annotation import BoxGroup
import os.path

HOST = str(
    subprocess.run(
        "/sbin/ip route|awk '/default/ { print $3 }'", shell=True, capture_output=True
    ).stdout.strip(),
    "ascii",
)
PORT = 8080
URL = f"http://{HOST}:{PORT}/invocations"


class TestInvocations(TestCase):
    def setUp(self) -> None:
        self.page_images = [
            Image.open(os.path.join(os.path.dirname(__file__), "data", "page0.png")).tobase64()
        ]

    def test__invoke(self):
        request = {"instances": [{"images": self.page_images}]}

        resp = requests.post(URL, json=request)

        self.assertEqual(resp.status_code, 200)

        box_groups = [BoxGroup.from_json(p) for p in resp.json()["predictions"][0]]

        box_types = set(bg.type for bg in box_groups)
        self.assertEqual(box_types, {"Title", "Text"})

    def test_jsonl_invocation(self):
        doc = {"images": self.page_images}
        docs = [doc, doc]
        request = "\n".join([json.dumps(d) for d in docs])

        resp = requests.post(
            f"{URL}",
            data=request,
            headers={
                "content-type": "application/jsonlines",
                "accept": "application/jsonlines",
            },
        )
        self.assertEqual(resp.status_code, 200)
        for line in resp.text.split("\n"):
            pred = json.loads(line)
            box_groups = [BoxGroup.from_json(sg) for sg in pred]
            box_types = set(bg.type for bg in box_groups)
            self.assertEqual({"Title", "Text"}, box_types)
