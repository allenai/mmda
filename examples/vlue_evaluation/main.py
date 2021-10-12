"""Compare VILA predictors to other models on VLUE."""
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from mmda.eval.vlue import PredictedDoc, read_labels, score
from mmda.parsers.grobid_parser import GrobidHeaderParser
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.predictors.hf_predictors.vila_predictor import (
    BaseVILAPredictor,
    HVILAPredictor,
    IVILAPredictor,
)
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.types.box import Box
from mmda.types.document import Document
from vila.pdftools.pdf_extractor import PDFExtractor


@dataclass
class VluePrediction(PredictedDoc):
    """Conforms to PredictedDoc protocol."""

    id: str  # pylint: disable=invalid-name
    title: str
    abstract: str


# Misc. required things for VILA document conversion code
SS_PARSER = SymbolScraperParser(sscraper_bin_path="")  # A dummy ssparser
PDF_EXTRACTOR = PDFExtractor("pdfplumber")


def _to_box(textblock, page_id) -> Box:
    block = textblock.block
    return Box(block.x_1, block.y_1, block.width, block.height, page_id)


def _convert_to_doc_object(pdf_tokens, pdf_images=None):
    page_to_row_to_tokens = defaultdict(lambda: defaultdict(list))

    for page_id, page_tokens in enumerate(pdf_tokens):
        for line_id, line_tokens in enumerate(page_tokens.get_text_segments()):
            for _token_id, token in enumerate(line_tokens):
                page_to_row_to_tokens[page_id][line_id].append(
                    {
                        "text": token.text,
                        "bbox": _to_box(token, page_id),
                    }
                )

    doc_dict = (
        SS_PARSER._convert_nested_text_to_doc_json(  # pylint: disable=protected-access
            {
                page: dict(row_to_tokens.items())
                for page, row_to_tokens in page_to_row_to_tokens.items()
            }
        )
    )

    doc = Document.from_json(doc_dict=doc_dict)
    if pdf_images is not None:
        doc.images = pdf_images
    return doc


def _load_pdf(path_to_pdf: str):
    pdf_tokens, pdf_images = PDF_EXTRACTOR.load_tokens_and_image(
        path_to_pdf,
    )
    doc = _convert_to_doc_object(pdf_tokens, pdf_images)
    return doc


def vila_prediction(
    id_: str,
    doc: Document,
    vila_predictor: BaseVILAPredictor,  # pylint: disable=redefined-outer-name
    layout_regions_predictor: LayoutParserPredictor,
    equation_layout_regions_predictor: Optional[LayoutParserPredictor] = None,
) -> VluePrediction:
    # Predict regions
    layout_regions = layout_regions_predictor.predict(doc)
    equation_layout_regions = (
        equation_layout_regions_predictor.predict(doc)
        if equation_layout_regions_predictor
        else []
    )
    doc.annotate(blocks=layout_regions + equation_layout_regions)

    # Predict token types
    span_groups = vila_predictor.predict(doc)
    doc.annotate(preds=span_groups)

    title = []
    abstract = []

    for span_group in doc.preds:
        if span_group.type == 1:
            title.append(span_group)
        elif span_group.type == 9:
            abstract.append(span_group)

    title = " ".join([" ".join(x.symbols) for x in title])
    abstract = "\n".join([" ".join(x.symbols) for x in abstract])

    return VluePrediction(id=id_, title=title, abstract=abstract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdfs-basedir", type=str, nargs="?", required=True)
    parser.add_argument("--labels-json-path", type=str, nargs="?", required=True)
    parser.add_argument("--vila-type", type=str, nargs="?", required=True)
    parser.add_argument("--vila-model-path", type=str, nargs="?", required=True)
    args = parser.parse_args()

    def pdf_path(id_: str) -> str:
        return os.path.join(args.pdfs_basedir, id_, "{}.pdf".format(id_))

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

    title_scores = {"vila": []}
    abstract_scores = {"vila": []}
    labels = read_labels(args.labels_json_path)

    for label in labels:
        # Known failing PDFs are excluded ...
        if label.id in [
            "396fb2b6ec96ff74e22ddd2484a9728257cccfbf",
            "3ef6e51baee01b4c90c188a964f2298b7c309b07",
            "4277d1ec41d88d595a0d80e4ab4146d8c2db2539",
            "564a73c07436e1bd75e31b54825d2ba8e4fb68b7",
        ]:
            continue

        item_pdf_path = pdf_path(label.id)
        vila_doc = _load_pdf(item_pdf_path)

        try:
            vila_pred = vila_prediction(
                label.id,
                vila_doc,
                vila_predictor=vila_predictor,
                layout_regions_predictor=layout_predictor,
                equation_layout_regions_predictor=equation_layout_predictor,
            )

            title_scores["vila"].append(score(label, vila_pred, "title"))
            abstract_scores["vila"].append(score(label, vila_pred, "abstract"))
        except (AssertionError, IndexError) as ex:
            print("Unable to process PDF: {}!", label.id)

    print("-------- TITLE --------")
    print(
        "VILA---\nMean: {}; Min: {}; Max: {}".format(
            mean(title_scores["vila"]),
            min(*title_scores["vila"]),
            max(*title_scores["vila"]),
        )
    )

    print("-------- ABSTRACT --------")
    print(
        "VILA---\nMean: {}; Min: {}; Max: {}".format(
            mean(abstract_scores["vila"]),
            min(*abstract_scores["vila"]),
            max(*abstract_scores["vila"]),
        )
    )
