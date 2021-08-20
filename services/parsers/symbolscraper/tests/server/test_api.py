import json
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import BaseModel

from server.api import make_app


class FakeInstance(BaseModel):
    field1: List[int]
    field2: str


class FakePrediction(BaseModel):
    output: List[str]


class MockPredictor:
    def __init__(self, *_):
        self.predict_batch_call_count = 0

    def predict_batch(self, instances: List[FakeInstance]) -> List[FakePrediction]:
        self.predict_batch_call_count += 1
        return [
            FakePrediction(output=[f1 * instance.field2 for f1 in instance.field1])
            for instance in instances
        ]


def get_mock_predictor():
    return MockPredictor()


TEST_INSTANCES = [
    {"field1": [1, 2, 3], "field2": "some\nstr"},
    {"field1": [2, 1], "field2": "someotherstr"},
]

TEST_PREDICTIONS = [
    {"output": ["some\nstr", "some\nstrsome\nstr", "some\nstrsome\nstrsome\nstr"]},
    {"output": ["someotherstrsomeotherstr", "someotherstr"]},
]


@patch("server.api.Instance", FakeInstance)
@patch("server.api.Prediction", FakePrediction)
@patch("server.api.get_predictor", get_mock_predictor)
class TestApi(unittest.TestCase):
    def test_invocations_jsonl__rejects_json_paired_with_jsonl(self):
        client = TestClient(make_app())

        response1 = client.post(
            "/invocations_jsonl",
            data="",
            headers={
                "Content-Type": "application/jsonlines",
                "Accept": "application/json",
            },
        )
        response2 = client.post(
            "/invocations_jsonl",
            data="",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/jsonlines",
            },
        )

        self.assertEqual(response1.status_code, 400)
        self.assertEqual(response2.status_code, 400)

    def test_invocations__rejects_json_paired_with_jsonl(self):
        client = TestClient(make_app())

        response1 = client.post(
            "/invocations",
            data="",
            headers={
                "Content-Type": "application/jsonlines",
                "Accept": "application/json",
            },
        )
        response2 = client.post(
            "/invocations",
            data="",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/jsonlines",
            },
        )

        self.assertEqual(response1.status_code, 400)
        self.assertEqual(response2.status_code, 400)

    def test_invocations__handles_json_requests(self):
        client = TestClient(make_app())

        request_json = {"instances": TEST_INSTANCES}

        response = client.post(
            "/invocations",
            json=request_json,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        expected_response = {"predictions": TEST_PREDICTIONS}

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected_response)

    def test_invocations__handles_jsonlines_requests(self):
        client = TestClient(make_app())

        request_jsonlines = "\n".join(
            [json.dumps(instance) for instance in TEST_INSTANCES]
        )

        response = client.post(
            "/invocations",
            data=request_jsonlines,
            headers={
                "Content-Type": "application/jsonlines",
                "Accept": "application/jsonlines",
            },
        )

        expected_response = "\n".join([json.dumps(pred) for pred in TEST_PREDICTIONS])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), expected_response)

    def test_invocations_jsonl__handles_jsonlines_requests(self):
        client = TestClient(make_app())

        request_jsonlines = "\n".join(
            [json.dumps(instance) for instance in TEST_INSTANCES]
        )

        response = client.post(
            "/invocations_jsonl",
            data=request_jsonlines,
            headers={
                "Content-Type": "application/jsonlines",
                "Accept": "application/jsonlines",
            },
        )

        expected_response = "\n".join([json.dumps(pred) for pred in TEST_PREDICTIONS])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), expected_response)


@patch("server.api.Instance", FakeInstance)
@patch("server.api.Prediction", FakePrediction)
class TestBatchSizing(unittest.TestCase):
    def _test_predict_batch_calls(
        self,
        instances: List[Dict[str, Any]],
        expected_call_count: int,
        batch_size: Optional[int] = None,
    ):
        mock_predictor = MockPredictor()

        def _get_mock_predictor():
            return mock_predictor

        with patch("server.api.get_predictor", _get_mock_predictor):
            if batch_size is not None:
                client = TestClient(make_app(batch_size))
            else:
                client = TestClient(make_app())

            request_json = {"instances": instances}
            resp = client.post(
                "/invocations",
                json=request_json,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            self.assertTrue(resp.status_code, 200)
            self.assertEqual(
                mock_predictor.predict_batch_call_count, expected_call_count
            )

    def test_batch_size__defaults_to_1(self):
        self._test_predict_batch_calls(TEST_INSTANCES, 2)

    def test_batch_size__respects_desired_batch_size(self):
        self._test_predict_batch_calls(TEST_INSTANCES, 1, batch_size=2)
        self._test_predict_batch_calls(TEST_INSTANCES * 2, 2, batch_size=2)
        self._test_predict_batch_calls(TEST_INSTANCES * 2, 2, batch_size=3)
        self._test_predict_batch_calls(TEST_INSTANCES * 5, 3, batch_size=4)

    def test_batch_size__rejects_non_positive_ints(self):
        with self.assertRaises(ValueError):
            make_app(batch_size=0)

        with self.assertRaises(ValueError):
            make_app(batch_size=-1)
