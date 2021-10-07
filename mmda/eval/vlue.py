import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Optional, Protocol

from mmda.eval import s2
from mmda.eval.metrics import levenshtein
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


@dataclass(frozen=True)
class LabeledDoc:
    id: str
    title: str
    abstract: str
    url: str


class PredictedDoc(Protocol):
    @property
    def title(self):
        raise NotImplementedError

    @property
    def abstract(self):
        raise NotImplementedError


@dataclass
class DefaultPredictedDoc:
    id: str
    title: str
    abstract: str


# Misc. required things for VILA document conversion code
_ssparser = SymbolScraperParser(sscraper_bin_path="")  # A dummy ssparser
_pdf_extractor = PDFExtractor("pdfplumber")


def _to_Box(textblock, page_id, page_with, page_height):
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
                        "bbox": _to_Box(
                            token, page_id, page_tokens.width, page_tokens.height
                        ),
                    }
                )

    doc_dict = _ssparser._convert_nested_text_to_doc_json(
        {
            page: {row: tokens for row, tokens in row_to_tokens.items()}
            for page, row_to_tokens in page_to_row_to_tokens.items()
        }
    )

    doc = Document.from_json(doc_dict=doc_dict)
    if pdf_images is not None:
        doc.images = pdf_images
    return doc


def _load_pdf(path_to_pdf):
    pdf_tokens, pdf_images = _pdf_extractor.load_tokens_and_image(
        path_to_pdf,
    )
    doc = _convert_to_doc_object(pdf_tokens, pdf_images)
    return doc


def grobid_prediction(pdf_path: str, parser: GrobidHeaderParser) -> PredictedDoc:
    doc = parser.parse(pdf_path)

    title = " ".join(doc.title[0].symbols)
    abstract = "\n".join([" ".join(x.symbols) for x in doc.abstract])

    return DefaultPredictedDoc(id=pdf_path, title=title, abstract=abstract)


def s2_prediction(id: str) -> PredictedDoc:
    metadata = s2.get_paper_metadata(id)

    title = metadata.title if metadata.title else ""
    abstract = metadata.abstract if metadata.abstract else ""

    return DefaultPredictedDoc(id=id, title=title, abstract=abstract)


def vila_prediction(
    id: str,
    doc: Document,
    vila_predictor: BaseVILAPredictor,
    layout_regions_predictor: LayoutParserPredictor,
    equation_layout_regions_predictor: Optional[LayoutParserPredictor] = None,
) -> PredictedDoc:
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

    return DefaultPredictedDoc(id=id, title=title, abstract=abstract)


def read_labels(labels_json_path: str) -> list[LabeledDoc]:
    with open(labels_json_path) as f:
        labels = [LabeledDoc(**l) for l in json.loads(f.read())]

    return labels


def score(label: LabeledDoc, pred: PredictedDoc, attr: str) -> float:
    a = label.__getattribute__(attr)
    b = pred.__getattribute__(attr)

    return 1 - levenshtein(a, b, case_sensitive=True) / max(len(a), len(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels-json-path", type=str, nargs="?", required=True)
    parser.add_argument("--vila-type", type=str, nargs="?", required=True)
    parser.add_argument("--vila-model-path", type=str, nargs="?", required=True)
    args = parser.parse_args()

    labels = read_labels(args.labels_json_path)
    source_pdf_path = "./resources/s2_vlue/original-pawls-annotation/{}/{}.pdf"

    grobid = GrobidHeaderParser()

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

    title_scores = {"grobid": [], "s2": [], "vila": []}
    abstract_scores = {"grobid": [], "s2": [], "vila": []}

    for label in labels:
        print(label.id)
        if label.id in [
            "396fb2b6ec96ff74e22ddd2484a9728257cccfbf",
            "3ef6e51baee01b4c90c188a964f2298b7c309b07",
            "4277d1ec41d88d595a0d80e4ab4146d8c2db2539",
            "564a73c07436e1bd75e31b54825d2ba8e4fb68b7",
        ]:
            continue

        item_pdf_path = source_pdf_path.format(label.id, label.id)

        grobid_pred = grobid_prediction(item_pdf_path, grobid)
        title_scores["grobid"].append(score(label, grobid_pred, "title"))
        abstract_scores["grobid"].append(score(label, grobid_pred, "abstract"))

        s2_pred = s2_prediction(label.id)
        title_scores["s2"].append(score(label, s2_pred, "title"))
        abstract_scores["s2"].append(score(label, s2_pred, "abstract"))

        # vila_doc = _load_pdf(item_pdf_path)
        # try:
        #    vila_pred = vila_prediction(
        #        label.id,
        #        vila_doc,
        #        vila_predictor=vila_predictor,
        #        layout_regions_predictor=layout_predictor,
        #        equation_layout_regions_predictor=equation_layout_predictor,
        #    )

        #    title_scores["vila"].append(score(label, vila_pred, "title"))
        #    abstract_scores["vila"].append(score(label, vila_pred, "abstract"))
        # except (AssertionError, IndexError) as ex:
        #    print("Unable to process PDF: {}!", label.id)

    print("-------- TITLE --------")
    print(
        "GROBID---\nMean: {}; Min: {}; Max: {}".format(
            mean(title_scores["grobid"]),
            min(*title_scores["grobid"]),
            max(*title_scores["grobid"]),
        )
    )
    print(
        "S2---\nMean: {}; Min: {}; Max: {}".format(
            mean(title_scores["s2"]),
            min(*title_scores["s2"]),
            max(*title_scores["s2"]),
        )
    )
    # print(
    #    "VILA---\nMean: {}; Min: {}; Max: {}".format(
    #        mean(title_scores["vila"]),
    #        min(*title_scores["vila"]),
    #        max(*title_scores["vila"]),
    #    )
    # )

    print("-------- ABSTRACT --------")
    print(
        "GROBID---\nMean: {}; Min: {}; Max: {}".format(
            mean(abstract_scores["grobid"]),
            min(*abstract_scores["grobid"]),
            max(*abstract_scores["grobid"]),
        )
    )
    print(
        "S2---\nMean: {}; Min: {}; Max: {}".format(
            mean(abstract_scores["s2"]),
            min(*abstract_scores["s2"]),
            max(*abstract_scores["s2"]),
        )
    )
    # print(
    #    "VILA---\nMean: {}; Min: {}; Max: {}".format(
    #        mean(abstract_scores["vila"]),
    #        min(*abstract_scores["vila"]),
    #        max(*abstract_scores["vila"]),
    #    )
    # )
