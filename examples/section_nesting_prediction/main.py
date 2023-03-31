"""
Tests for SectionNestingPredictor

@rauthur
"""

import pathlib
import unittest
from copy import deepcopy

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.xgb_predictors.section_nesting_predictor import (
    SectionNestingPredictor,
)
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import SpanGroup


def main():
    file_path = pathlib.Path(__file__).parent
    model_path = file_path / "nesting.bin"
    pdf_path = file_path / "sample.pdf"

    parser = PDFPlumberParser(extra_attrs=[])
    rasterizer = PDF2ImageRasterizer()
    layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/PubLayNet"
    )
    vila = IVILAPredictor.from_pretrained(
        "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        agg_level="row",
        added_special_sepration_token="[BLK]",
    )

    predictor = SectionNestingPredictor(model_path)

    doc = parser.parse(pdf_path)
    images = rasterizer.rasterize(pdf_path, dpi=72)
    doc.annotate_images(images)

    blocks = layout_predictor.predict(doc)
    doc.annotate(blocks=blocks)

    results = vila.predict(doc)
    doc.annotate(results=results)

    # Extract sections from VILA predictions and re-add boxes
    vila_sections = []

    for i, span_group in enumerate(doc.results):
        if span_group.type != 4:
            continue

        # Boxes are not returned from VILA so reach into tokens
        token_spans = [deepcopy(t.spans) for t in span_group.tokens]
        token_spans = [span for l in token_spans for span in l]  # Flatten list

        # Maintain the text from VILA for each span group but replace newlines
        metadata = deepcopy(span_group.metadata)
        metadata.text = span_group.text.replace("\n", " ")

        vila_sections.append(
            SpanGroup(
                spans=token_spans,
                box_group=deepcopy(span_group.box_group),
                id=i,  # Ensure some ID is created
                doc=None,  # Allows calling doc.annotate(...)
                metadata=metadata,
            )
        )

    doc.annotate(sections=vila_sections)

    nestings = predictor.predict(doc)
    section_index = {s.id: s for s in nestings}

    for section in nestings:
        parent_id = section.metadata.parent_id

        if parent_id == -1:
            print(f"Section '{section.text}' is top-level!")
            continue

        parent = section_index[parent_id]
        print(f"Section '{section.text}' has parent '{parent.text}'")


if __name__ == "__main__":
    main()
