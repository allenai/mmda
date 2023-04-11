import json
from typing import Dict, List, Optional
import argparse
import os

from tqdm import tqdm
import torch
from layoutparser.elements import TextBlock, Rectangle, Layout
from layoutparser.visualization import draw_box
from PIL.Image import Image

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import SpanGroup
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor


DOCBANK_LABEL_MAP = {
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
    15: "Footnote"
}



def draw_tokens(
    image: Image,
    doc_tokens: List[SpanGroup],
    color_map: Optional[Dict[int, str]] = None,
    token_boundary_width: int = 0,
    alpha: float = 0.25,
    **lp_draw_box_kwargs,
):
    """Draw MMDA tokens as rectangles on the an image of a page."""

    w, h = image.size
    layout = [
        TextBlock(
            Rectangle(
                *token.spans[0]
                .box.get_absolute(  # pyright: ignore
                    page_height=h,
                    page_width=w
                ).coordinates
            ),
            type=token.type,
            text=token.text,
        )
        for token in doc_tokens
    ]

    return draw_box(
        image,
        Layout(blocks=layout),
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **lp_draw_box_kwargs,
    )


def draw_blocks(
    image: Image,
    doc_tokens: List[SpanGroup],
    color_map: Optional[Dict[int, str]] = None,
    token_boundary_width: int = 0,
    alpha: float = 0.25,
    **lp_draw_box_kwargs,
):

    w, h = image.size
    layout = [
        TextBlock(
            Rectangle(
                *token.box_group.boxes[0]   # pyright: ignore
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=token.type,
            text=token.text,
        )
        for token in doc_tokens
    ]
    return draw_box(
        image,
        Layout(blocks=layout),
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **lp_draw_box_kwargs,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, nargs="+")
    parser.add_argument("--rasterizer-dpi", type=int, default=72)
    parser.add_argument(
        "--vila-model-path",
        type=str,
        default="allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
    )
    parser.add_argument(
        "--export-folder",
        type=str,
        default="pdf-predictions",
    )
    args = parser.parse_args()

    # VILA predicts regions on pages, like paragraphs, tables, etc.
    # for more info, see https://api.semanticscholar.org/CorpusID:245704273
    vila_predictor = IVILAPredictor.from_pretrained(
        args.vila_model_path,
        agg_level="row",
        added_special_sepration_token="[BLK]",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # LayoutParser can be used for predicting many layout regions;
    # here, we use it to find equations. For more info, see
    # https://api.semanticscholar.org/CorpusID:232404723
    equation_layout_predictor = LayoutParserPredictor.from_pretrained(
        config_path="lp://efficientdet/MFD",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # these two parsers are used to extract text and images from PDFs
    pdfplumber_parser = PDFPlumberParser()
    rasterizer = PDF2ImageRasterizer()

    pbar = tqdm(args.pdf_path)
    for pdf_path in pbar:
        assert os.path.exists(pdf_path), f"PDF file {pdf_path} does not exist."

        pbar.set_description(f"Working on {pdf_path}")

        doc = pdfplumber_parser.parse(input_pdf_path=pdf_path)
        images = rasterizer.rasterize(
            input_pdf_path=pdf_path, dpi=args.rasterizer_dpi
        )
        doc.annotate_images(images)

        # Detect equation regions
        equation_layout_regions = equation_layout_predictor.predict(doc)
        doc.annotate(equations=equation_layout_regions)

        # Obtaining Textual Predictions
        spans = vila_predictor.predict(doc)
        doc.annotate(sections=spans)

        # get location where to save data
        save_folder = os.path.join(
            args.export_folder, os.path.basename(pdf_path).rstrip(".pdf")
        )
        os.makedirs(save_folder, exist_ok=True)

        # save images with layout and text predictions
        for pid, (page, image) in enumerate(zip(doc.pages, doc.images)):
            new_tokens = []
            for pred in page.sections:
                for token in pred.tokens:
                    new_token = token.from_json(token.to_json())
                    new_token.type = pred.type
                    new_token.text = token.text
                    new_tokens.append(new_token)

            # draw sections and equations
            viz = draw_tokens(image, new_tokens, alpha=0.6)
            viz = draw_blocks(viz, page.equations, alpha=0.2)

            viz.save(f"{save_folder}/{pid}.png")

        # Save all text alongside name of section it belongs to
        with open(f"{save_folder}/sections.jsonl", "w") as f:
            for pid, page in enumerate(doc.pages):
                for pred in page.sections:
                    data = {
                        "text": str(pred.text),
                        "type": DOCBANK_LABEL_MAP[int(pred.type)],
                        "page": int(pid),
                    }
                    f.write(f"{json.dumps(data, sort_keys=True)}\n")

        # Save all equations
        with open(f"{save_folder}/equations.jsonl", "w") as f:
            for pid, page in enumerate(doc.pages):
                for pred in page.equations:
                    data = {
                        "text": str(pred.text),
                        "page": int(pid),
                    }
                    f.write(f"{json.dumps(data, sort_keys=True)}\n")
