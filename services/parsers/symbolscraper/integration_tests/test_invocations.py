import base64
import pathlib
import subprocess
from unittest import TestCase

import requests

from model.prediction import Prediction


HOST = str(
    subprocess.run(
        "/sbin/ip route|awk '/default/ { print $3 }'", shell=True, capture_output=True
    ).stdout.strip(),
    "ascii",
)
PORT = 8080
URL = f"http://{HOST}:{PORT}/invocations"

TEST_PDF = f"{str(pathlib.Path(__file__).parent)}/data/arxiv-1906.08632.pdf"
TEST_IMAGE = f"{str(pathlib.Path(__file__).parent)}/data/arxiv-1906.08632.png"


def get_test_request():
    with open(TEST_PDF, "rb") as f:
        pdf_bytes = str(base64.b64encode(f.read()), "ascii")
    return {"instances": [{"pdf": pdf_bytes}]}


class TestInvocations(TestCase):
    def test__live_invocation(self):
        request = get_test_request()
        resp = requests.post(URL, json=request)
        self.assertEqual(resp.status_code, 200)
        # test that i can rehydrate the resp
        pred = Prediction(**resp.json()["predictions"][0])
        with open(TEST_IMAGE, "rb") as f:
            image = str(base64.b64encode(f.read()), "ascii")
        assert pred.images[0] == image
