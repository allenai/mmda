import enum
import json
import time
from typing import Iterable, List, Optional, Dict

from mmda.types.document import Document as MmdaDocument
import uvicorn

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.api_model import SpanGroup
from server.predictor import Predictor, PredictorConfig


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
        instances: List[Dict]

    class InvocationsResponse(BaseModel):
        predictions: List[List[SpanGroup]]

    app = FastAPI()

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
        instances = [json.loads(line) for line in lines]
        predictions = perform_invocations(instances, predictor).predictions

        jsonl_resp = "\n".join(
            json.dumps([p.dict() for p in pred]) for pred in predictions
        )

        return PlainTextResponse(jsonl_resp, media_type=ContentType.JSONLINES.value)

    @app.post("/invocations", response_model=InvocationsResponse)
    async def invocations(
        req: InvocationsRequest, predictor: Predictor = Depends(get_predictor)
    ) -> InvocationsResponse:
        resp = perform_invocations(req.instances, predictor)
        return resp

    def perform_invocations(
        instances: List[MmdaDocument], predictor: Predictor
    ) -> InvocationsResponse:
        predictions = []

        for batch in _batchify(instances, batch_size):
            mmda_docs = [MmdaDocument.from_json(d) for d in batch]
            mmda_predictions = predictor.predict_batch(mmda_docs)
            api_predictions = [
                [SpanGroup.from_mmda(sg) for sg in batch] for batch in mmda_predictions
            ]
            predictions.extend(api_predictions)

        return InvocationsResponse(predictions=predictions)

    @app.get("/ping")
    async def health_check():
        return {"message": "Ok"}

    return app


def _batchify(instances: Iterable[Dict], batch_size: int) -> Iterable[List[Dict]]:
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


# For local development
# To run production server, use entrypoint.py
if __name__ == "__main__":
    app = make_app()
    uvicorn.run(app, port=8080)
