from collections import defaultdict
import argparse
from copy import copy
import os

from tqdm import tqdm
import layoutparser as lp
from vila.pdftools.pdf_extractor import PDFExtractor

from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.types.document import Document
from mmda.types.box import Box
from mmda.types.span import Span
from mmda.types.annotation import SpanGroup
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor, HVILAPredictor


ssparser = SymbolScraperParser(sscraper_bin_path="")  # A dummy ssparser
pdf_extractor = PDFExtractor("pdfplumber")

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


def _coordinates(box, canvas_width, canvas_height):
    return lp.Rectangle(box.l, box.t, (box.l + box.w), (box.t + box.h))


def draw_tokens(
    image, doc_tokens, color_map=None, token_boundary_width=0, alpha=0.25, **kwargs
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            _coordinates(token.spans[0].box, w, h),
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
        **kwargs
    )


def draw_blocks(
    image, doc_tokens, color_map=None, token_boundary_width=0, alpha=0.25, **kwargs
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            _coordinates(token.box_group.boxes[0], w, h),
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
        **kwargs
    )


def to_Box(textblock, page_id, page_with, page_height):
    block = textblock.block
    return Box(block.x_1, block.y_1, block.width, block.height, page_id)


def convert_to_doc_object(pdf_tokens, pdf_images=None):

    page_to_row_to_tokens = defaultdict(lambda: defaultdict(list))

    for page_id, page_tokens in enumerate(pdf_tokens):
        for line_id, line_tokens in enumerate(page_tokens.get_text_segments()):
            for token_id, token in enumerate(line_tokens):
                page_to_row_to_tokens[page_id][line_id].append(
                    {
                        "text": token.text,
                        "bbox": to_Box(
                            token, page_id, page_tokens.width, page_tokens.height
                        ),
                    }
                )

    doc_dict = ssparser._convert_nested_text_to_doc_json(
        {
            page: {row: tokens for row, tokens in row_to_tokens.items()}
            for page, row_to_tokens in page_to_row_to_tokens.items()
        }
    )

    doc = Document.from_json(doc_dict=doc_dict)
    if pdf_images is not None:
        doc.images = pdf_images
    return doc


def load_pdf(path_to_pdf):
    pdf_tokens, pdf_images = pdf_extractor.load_tokens_and_image(
        path_to_pdf,
    )
    doc = convert_to_doc_object(pdf_tokens, pdf_images)
    return doc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, nargs="+")
    parser.add_argument("--vila-type", type=str, default="ivila")
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
    
    pbar = tqdm(args.pdf_path)
    for pdf_path in pbar:
        pbar.set_description(f"Working on {pdf_path}")
        doc = load_pdf(pdf_path)

        # Obtaining Layout Predictions
        layout_regions = layout_predictor.predict(doc) # Detect content regions like paragraphs
        equation_layout_regions = equation_layout_predictor.predict(doc) # Detect equation regions 

        doc.annotate(blocks=layout_regions + equation_layout_regions)

        # Obtaining Textual Predictions
        spans = vila_predictor.predict(doc)
        doc.annotate(preds=spans)

        save_folder = os.path.join(args.export_folder, os.path.basename(pdf_path).split(".")[0])
        os.makedirs(save_folder)

        for pid in range(len(doc.pages)):
            
            new_tokens = []
            for pred in doc.pages[pid].preds:
                for token in pred.tokens:
                    _token = copy(token)
                    _token.type = DOCBANK_LABEL_MAP[pred.type]
                    new_tokens.append(
                        _token
                    )
                    
            viz = draw_blocks(doc.images[pid], doc.pages[pid].blocks, alpha=0)
            viz = draw_tokens(viz, new_tokens, alpha=0.6)

            viz.save(f"{save_folder}/{pid}.png")