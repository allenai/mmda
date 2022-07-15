import gradio as gr
from copy import copy
import torch
import torchvision
from torchvision import transforms
import requests
from PIL import Image

from main import (
    draw_blocks,
    draw_tokens,
    PDFPlumberParser,
    SpanGroup,
    LayoutParserPredictor,
    IVILAPredictor,
    DOCBANK_LABEL_MAP,
)


pdfplumber_parser = PDFPlumberParser()
vila_predictor = IVILAPredictor.from_pretrained(
    "docbank/layout_indicator-BLK-block/microsoft-layoutlm-base-uncased/",
    added_special_sepration_token="[BLK]",
    agg_level="block",
)
layout_predictor = LayoutParserPredictor.from_pretrained("lp://efficientdet/PubLayNet")
equation_layout_predictor = LayoutParserPredictor.from_pretrained(
    "lp://efficientdet/MFD"
)


def predict(pdf_path):
    doc = pdfplumber_parser.parse(input_pdf_path=pdf_path.name, load_images=True)
    layout_regions = layout_predictor.predict(
        doc
    )  # Detect content regions like paragraphs
    equation_layout_regions = equation_layout_predictor.predict(
        doc
    )  # Detect equation regions

    doc.annotate(blocks=layout_regions + equation_layout_regions)

    # Obtaining Textual Predictions
    spans = vila_predictor.predict(doc)
    doc.annotate(preds=spans)

    all_viz = []
    for pid in range(len(doc.pages)):

        new_tokens = []
        for pred in doc.pages[pid].preds:
            for token in pred.tokens:
                _token = copy(token)
                _token.type = DOCBANK_LABEL_MAP[pred.type]
                new_tokens.append(_token)

        viz = draw_blocks(doc.images[pid], doc.pages[pid].blocks, alpha=0)
        viz = draw_tokens(viz, new_tokens, alpha=0.6)
        # viz = draw_tokens(doc.images[pid], new_tokens, alpha=0.6)

        # w, h = doc.images[pid].size
        # bbox = [
        #     (
        #         token.type,
        #         *token.box_group.boxes[0]
        #         .get_absolute(page_height=h, page_width=w)
        #         .coordinates,
        #     )
        #     for token in doc.pages[pid].blocks
        # ]
        all_viz.append(viz)
    return all_viz


if __name__ == "__main__":
    
    iface = gr.Interface(
        fn=predict,
        inputs="file",
        outputs=gr.outputs.Carousel(gr.outputs.Image(
            type="pil", label="Output Image", labeled_segments=False
        ), label="Disease"),
        title=""
    ).launch()

    iface.test_launch()
    iface.launch()