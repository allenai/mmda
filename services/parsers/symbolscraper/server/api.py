import enum
import json
import time
from typing import Iterable, List, Optional

from fastapi import Depends, HTTPException, FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from model.instance import Instance
from model.prediction import Prediction
from model.predictor import Predictor, PredictorConfig


# For compatibility with AWS SageMaker,
# which prevents inclusion of adhoc response headers, see:
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html#API_runtime_InvokeEndpoint_ResponseSyntax.
# Limit: 1024 US-ASCII characters.
CUSTOM_ATTRIBUTES_HEADER_NAME = "X-Amzn-SageMaker-Custom-Attributes"


_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor

    if not _predictor:
        config = PredictorConfig()
        _predictor = Predictor(config=config)

    return _predictor


class ContentType(enum.Enum):
    JSON = "application/json"
    JSONLINES = "application/jsonlines"


def make_app(batch_size: int = 1):
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("Batch size must be positive integer")

    class InvocationsRequest(BaseModel):
        """Represents the JSON body of a set of inference requests."""

        instances: List[Instance] = Field(
            description="A list of Instances over which to perform inference"
        )

    class InvocationsResponse(BaseModel):
        """The results of inference over each passed instance"""

        predictions: List[Prediction] = Field(description="The predictions")

    class JsonResponse(Response):
        media_type = "application/json"

        def render(self, content) -> bytes:
            """
            This is copied from starlette.responses.JSONResponse
            with allow_nan=False removed.
            """
            return json.dumps(
                content,
                ensure_ascii=False,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")

    app = FastAPI(default_response_class=JsonResponse)

    @app.middleware("http")
    async def add_request_processing_time(request: Request, call_next):
        """Records total time (seconds) processing a request"""
        start_time = time.monotonic()
        response = await call_next(request)
        processing_time = time.monotonic() - start_time

        response.headers[CUSTOM_ATTRIBUTES_HEADER_NAME] = json.dumps(
            {"processing_time": processing_time}
        )

        return response

    @app.middleware("http")
    async def jsonl_invocations_handler(request: Request, call_next):
        """Intercepts JSONLINES requests and routes to appropriate handler."""
        accept = request.headers.get("accept")
        content_type = request.headers.get("content-type")

        if (
            content_type == ContentType.JSONLINES.value
            and accept == ContentType.JSONLINES.value
        ):
            scope = request.scope

            if not scope["path"].endswith("_jsonl"):
                scope["path"] += "_jsonl"
                new_request = Request(scope, request.receive)
                return await call_next(new_request)

        elif (
            content_type == ContentType.JSONLINES.value
            or accept == ContentType.JSONLINES.value
        ):
            return Response(
                content=f"Must pair {ContentType.JSONLINES.value} content-type with {ContentType.JSONLINES.value} accept",
                status_code=400,
            )

        return await call_next(request)

    @app.post("/invocations_jsonl")
    async def invocations_jsonl(
        req: Request, predictor: Predictor = Depends(get_predictor)
    ) -> Response:
        body_text = (await req.body()).decode()
        lines = [line.strip() for line in body_text.split("\n") if line.strip()]
        instances = [Instance.parse_raw(line) for line in lines]
        json_req = InvocationsRequest(instances=instances)
        json_resp = perform_invocations(json_req, predictor)

        jsonl_resp = "\n".join(
            [prediction.json() for prediction in json_resp.predictions]
        )

        return PlainTextResponse(jsonl_resp, media_type=ContentType.JSONLINES.value)

    @app.post("/invocations")
    async def invocations(
        req: InvocationsRequest, predictor: Predictor = Depends(get_predictor)
    ) -> InvocationsResponse:
        resp = perform_invocations(req, predictor)
        return resp

    def perform_invocations(
        req: InvocationsRequest, predictor: Predictor
    ) -> InvocationsResponse:
        predictions = []

        for batch in _batchify(req.instances, batch_size):
            prediction_batch = predictor.predict_batch(batch)
            predictions.extend(prediction_batch)

        return InvocationsResponse(predictions=predictions)

    @app.get("/ping")
    async def health_check():
        return {"message": "Okalee-dokalee"}

    return app


def _batchify(
    instances: Iterable[Instance], batch_size: int
) -> Iterable[List[Instance]]:
    current_batch = []

    for instance in instances:
        current_batch.append(instance)

        if len(current_batch) == batch_size:
            to_yield = current_batch
            current_batch = []
            yield to_yield

    if current_batch:
        yield current_batch


def initialize_api(batch_size: int = 1):
    get_predictor()
    return make_app(batch_size=batch_size)
