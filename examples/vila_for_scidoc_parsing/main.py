from typing import List
import argparse
from copy import copy
import os

from tqdm import tqdm
import layoutparser as lp

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import Entity
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor, HVILAPredictor


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


def draw_tokens(
    image,
    doc_tokens: List[Entity],
    color_map=None,
    token_boundary_width=0,
    alpha=0.25,
    **kwargs,
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            lp.Rectangle(
                *token.spans[0]
                .box.get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=token.type,
            text=token.symbols[0],
        )
        for token in doc_tokens
    ]
    return lp.draw_box(
        image,
        layout,
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **kwargs,
    )


def draw_blocks(
    image,
    doc_tokens: List[Entity],
    color_map=None,
    token_boundary_width=0,
    alpha=0.25,
    **kwargs,
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            lp.Rectangle(
                *token.box_group.boxes[0]
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=token.box_group.type,
            text=token.symbols[0],
        )
        for token in doc_tokens
    ]
    return lp.draw_box(
        image,
        layout,
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **kwargs,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, nargs="+")
    parser.add_argument("--vila-type", type=str, default="ivila")
    parser.add_argument("--rasterizer-dpi", type=int, default=72)
    parser.add_argument(
        "--vila-model-path",
        type=str,
        default="docbank/layout_indicator-BLK-block/microsoft-layoutlm-base-uncased/",
    )
    parser.add_argument(
        "--export-folder",
        type=str,
        default="pdf-predictions",
    )

    args = parser.parse_args()

    if args.vila_type == "ivila":
        vila_predictor = IVILAPredictor.from_pretrained(
            args.vila_model_path,
            added_special_sepration_token="[BLK]",
            agg_level="block",
        )
    elif args.vila_type == "hvila":
        vila_predictor = HVILAPredictor.from_pretrained(
            args.vila_model_path, agg_level="block", group_bbox_agg="first"
        )

    layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/PubLayNet"
    )
    equation_layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/MFD"
    )

    pdfplumber_parser = PDFPlumberParser()
    rasterizer = PDF2ImageRasterizer()

    pbar = tqdm(args.pdf_path)
    for pdf_path in pbar:
        pbar.set_description(f"Working on {pdf_path}")

        doc = pdfplumber_parser.parse(input_pdf_path=pdf_path)
        images = rasterizer.rasterize(input_pdf_path=pdf_path, dpi=args.rasterizer_dpi)
        doc.annotate_images(images)

        # Obtaining Layout Predictions
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

        save_folder = os.path.join(
            args.export_folder, os.path.basename(pdf_path).rstrip(".pdf")
        )
        os.makedirs(save_folder, exist_ok=True)

        for pid in range(len(doc.pages)):

            new_tokens = []
            for pred in doc.pages[pid].preds:
                for token in pred.tokens:
                    _token = copy(token)
                    _token.type = DOCBANK_LABEL_MAP[pred.type]
                    new_tokens.append(_token)

            viz = draw_blocks(doc.images[pid], doc.pages[pid].blocks, alpha=0)
            viz = draw_tokens(viz, new_tokens, alpha=0.6)

            viz.save(f"{save_folder}/{pid}.png")
