from typing import List
import argparse
from copy import copy
import os

from tqdm import tqdm
import layoutparser as lp

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import SpanGroup
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor, HVILAPredictor
from mmda.predictors.hf_predictors.token_classification_predictor import (
    IVILATokenClassificationPredictor,
    HVILATokenClassificationPredictor,
)


DOCBANK_LABEL_MAP = {
    "0": "paragraph",
    "1": "title",
    "2": "equation",
    "3": "reference",
    "4": "section",
    "5": "list",
    "6": "table",
    "7": "caption",
    "8": "author",
    "9": "abstract",
    "10": "footer",
    "11": "date",
    "12": "figure",
}
DOCBANK_LABEL_MAP = {int(key): val for key, val in DOCBANK_LABEL_MAP.items()}


def test_vila_predictors():
    layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/PubLayNet"
    )

    pdfplumber_parser = PDFPlumberParser()
    rasterizer = PDF2ImageRasterizer()

    doc = pdfplumber_parser.parse(input_pdf_path="tests/fixtures/1903.10676.pdf")
    images = rasterizer.rasterize(input_pdf_path="tests/fixtures/1903.10676.pdf", dpi=72)
    doc.annotate_images(images)

    layout_regions = layout_predictor.predict(doc)
    doc.annotate(blocks=layout_regions)

    ivilaA = IVILATokenClassificationPredictor.from_pretrained(
        "allenai/ivila-block-layoutlm-finetuned-docbank"
    )
    resA = ivilaA.predict(doc, subpage_per_run=2)
    del ivilaA

    ivilaB = IVILAPredictor.from_pretrained(
        "allenai/ivila-block-layoutlm-finetuned-docbank",
        agg_level="block",
        added_special_sepration_token="[BLK]",
    )

    resB = ivilaB.predict(doc)
    del ivilaB

    assert [ele.spans for ele in resA] == [ele.spans for ele in resB]
    assert [ele.type for ele in resA] == [DOCBANK_LABEL_MAP[ele.type] for ele in resB]

    hvilaA = HVILATokenClassificationPredictor.from_pretrained(
        "allenai/hvila-row-layoutlm-finetuned-docbank"
    )
    resA = hvilaA.predict(doc, subpage_per_run=2)
    del hvilaA

    hvilaB = HVILAPredictor.from_pretrained(
        "allenai/hvila-row-layoutlm-finetuned-docbank",
        agg_level="row",
        added_special_sepration_token="[BLK]",
    )
    resB = hvilaB.predict(doc)
    del hvilaB

    assert [ele.spans for ele in resA] == [ele.spans for ele in resB]
    assert [ele.type for ele in resA] == [DOCBANK_LABEL_MAP[ele.type] for ele in resB]