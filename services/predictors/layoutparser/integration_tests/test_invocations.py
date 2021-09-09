import subprocess
import requests
from unittest import TestCase
from mmda.types.image import Image
from model.prediction import Prediction, BoxGroup, Box
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
    def test__invoke(self):
        page_images = [
            Image.open(os.path.join(os.path.dirname(__file__), "data", "page0.png"))
        ]
        request = {"instances": [{"page_images": [i.tobase64() for i in page_images]}]}

        resp = requests.post(URL, json=request)

        self.assertEqual(resp.status_code, 200)

        predictions = [Prediction(**p) for p in resp.json()["predictions"]]

        self.assertTrue(len(predictions) > 0, "Empty predictions")
        box_types = set(g.type for p in predictions for g in p.groups)
        self.assertEqual(box_types, {"TextRegion", "SeparatorRegion"})
