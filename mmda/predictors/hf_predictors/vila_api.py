import os
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
from mmda.types.document import Document

from typing import Optional

app = Flask(__name__)

_predictor: Optional[IVILAPredictor] = None
_label_names = {
    0: "Title",
    1: "Author",
    2: "Abstract",
    3: "Keywords",
    4: "Section",
    5: "Paragraph",
    6: "List",
    7: "Bibliography",
    8: "Equation",
    9: "Algorithm",
    10: "Figure",
    11: "Table",
    12: "Caption",
    13: "Header",
    14: "Footer",
    15: "Footnote",
}


def init_predictors():
    global _predictor
    if not _predictor:
        _predictor = IVILAPredictor.from_pretrained(
            "/pipeline/vila-model",
            added_special_sepration_token="[BLK]",
            agg_level="row",
        )


@app.post("/")
def run_model():
    init_predictors()
    doc = Document.from_json(request.json)
    annotations = _predictor.predict(doc)
    for a in annotations:
        a.type = _label_names.get(a.type, f"label{a.type}")
    return jsonify(annotations)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
