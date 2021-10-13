import os
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

import mmda.predictors.lp_predictors as lpp
from mmda.types.document import Document

app = Flask(__name__)

_predictors = None

def init_predictors():
    global _predictors
    if not _predictors:
        _predictors = [
            lpp.LayoutParserPredictor.from_pretrained(path) for path in os.getenv("MODEL_CONFIG").split(",")
        ]


@app.post("/")
def run_model():
    init_predictors()
    doc = Document.from_json(request.json)
    doc_predictions = []
    for pred in _predictors:
        annotations = pred.predict(doc)
        doc_predictions.extend(annotations)
    return jsonify([p.to_json() for p in doc_predictions])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
