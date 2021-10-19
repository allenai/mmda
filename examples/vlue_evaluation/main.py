"""Compare VILA predictors to other models on VLUE."""
import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional

from mmda.eval.vlue import (
    LabeledDoc,
    PredictedDoc,
    grobid_prediction,
    random_prediction,
    read_labels,
    s2_prediction,
    score,
)
from mmda.parsers.grobid_parser import GrobidHeaderParser
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.predictors.hf_predictors.vila_predictor import (
    BaseVILAPredictor,
    HVILAPredictor,
    IVILAPredictor,
)
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
from vila.pdftools.pdf_extractor import PDFExtractor


@dataclass
class VluePrediction:
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


def _vila_docbank_extract_entities(types: List[str]):
    def extractor(doc: Document) -> Dict[str, List[SpanGroup]]:
        mapping = {
            "paragraph": 0,
            "title": 1,
            "equation": 2,
            "reference": 3,
            "section": 4,
            "list": 5,
            "table": 6,
            "caption": 7,
            "author": 8,
            "abstract": 9,
            "footer": 10,
            "date": 11,
            "figure": 12,
        }
        rmapping = {v: k for k, v in mapping.items()}

        int_types = set([mapping[x] for x in types])
        result = defaultdict(list)

        for span_group in doc.preds:
            if span_group.type in int_types:
                result[rmapping[span_group.type]].append(span_group)

        return result

    return extractor


def _vila_grotoap2_extract_entities(types: List[str]):
    def extractor(doc: Document) -> Dict[str, List[SpanGroup]]:
        # TODO: Have some sort of unified mapping between this and docbank
        # TODO: Below title and abstract have been lower-cased to match docbank
        mapping = {
            "BIB_INFO": 0,
            "REFERENCES": 1,
            "UNKNOWN": 2,
            "BODY_CONTENT": 3,
            "PAGE_NUMBER": 4,
            "TABLE": 5,
            "ACKNOWLEDGMENT": 6,
            "FIGURE": 7,
            "CONFLICT_STATEMENT": 8,
            "AFFILIATION": 9,
            "DATES": 10,
            "TYPE": 11,
            "title": 12,
            "AUTHOR": 13,
            "abstract": 14,
            "CORRESPONDENCE": 15,
            "EDITOR": 16,
            "COPYRIGHT": 17,
            "AUTHOR_TITLE": 18,
            "KEYWORDS": 19,
            "GLOSSARY": 20,
            "EQUATION": 21,
        }
        rmapping = {v: k for k, v in mapping.items()}

        int_types = set([mapping[x] for x in types])
        result = defaultdict(list)

        for span_group in doc.preds:
            if span_group.type in int_types:
                result[rmapping[span_group.type]].append(span_group)

        return result

    return extractor


def vila_prediction(
    id_: str,
    doc: Document,
    vila_predictor: BaseVILAPredictor,  # pylint: disable=redefined-outer-name
    vila_extractor: Callable[[Document], Dict[str, List[SpanGroup]]],
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

    extracted = vila_extractor(doc)
    title = " ".join([" ".join(x.symbols) for x in extracted["title"]])
    abstract = "\n".join([" ".join(x.symbols) for x in extracted["abstract"]])

    return VluePrediction(id=id_, title=title, abstract=abstract)


def _vila_models(model_name: str):
    if model_name == "ivila-block-layoutlm-finetuned-docbank":
        vila_predictor = IVILAPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-docbank",
            added_special_sepration_token="[BLK]",  # FIXME: typo in underlying repo
            agg_level="block",
        )
        vila_extractor = _vila_docbank_extract_entities(["title", "abstract"])

    elif model_name == "ivila-block-layoutlm-finetuned-grotoap2":
        vila_predictor = IVILAPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-grotoap2",
            added_special_sepration_token="[BLK]",
            agg_level="block",
        )
        vila_extractor = _vila_grotoap2_extract_entities(["title", "abstract"])

    elif model_name == "hvila-block-layoutlm-finetuned-docbank":
        vila_predictor = HVILAPredictor.from_pretrained(
            "allenai/hvila-block-layoutlm-finetuned-docbank",
            agg_level="block",
            added_special_sepration_token="[BLK]",
            group_bbox_agg="first",
        )
        vila_extractor = _vila_docbank_extract_entities(["title", "abstract"])

    elif model_name == "hvila-row-layoutlm-finetuned-docbank":
        vila_predictor = HVILAPredictor.from_pretrained(
            "allenai/hvila-row-layoutlm-finetuned-docbank",
            agg_level="row",
            added_special_sepration_token="[SEP]",
            group_bbox_agg="first",
        )
        vila_extractor = _vila_docbank_extract_entities(["title", "abstract"])

    elif model_name == "hvila-block-layoutlm-finetuned-grotoap2":
        vila_predictor = HVILAPredictor.from_pretrained(
            "allenai/hvila-block-layoutlm-finetuned-grotoap2",
            agg_level="block",
            added_special_sepration_token="[BLK]",
            group_bbox_agg="first",
        )
        vila_extractor = _vila_grotoap2_extract_entities(["title", "abstract"])

    elif model_name == "hvila-row-layoutlm-finetuned-grotoap2":
        vila_predictor = HVILAPredictor.from_pretrained(
            "allenai/hvila-row-layoutlm-finetuned-grotoap2",
            agg_level="row",
            added_special_sepration_token="[SEP]",
            group_bbox_agg="first",
        )
        vila_extractor = _vila_grotoap2_extract_entities(["title", "abstract"])

    return vila_predictor, vila_extractor


def save_prediction(
    writer: csv.DictWriter,
    label: LabeledDoc,
    pred: PredictedDoc,
    model: str,
    title_score: float,
    abstract_score: float,
) -> None:
    d = {
        "SHA": label.id,
        "URL": label.url,
        "Model": model,
        "Title": pred.title,
        "TitleScore": title_score,
        "Abstract": pred.abstract,
        "AbstractScore": abstract_score,
    }
    writer.writerow(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdfs-basedir", type=str, nargs="?", required=True)
    parser.add_argument("--labels-json-path", type=str, nargs="?", required=True)
    parser.add_argument("--output-csv-path", type=str, nargs="?", required=True)
    args = parser.parse_args()

    def pdf_path(id_: str) -> str:
        return os.path.join(args.pdfs_basedir, id_, "{}.pdf".format(id_))

    title_scores = defaultdict(list)
    abstract_scores = defaultdict(list)
    labels = read_labels(args.labels_json_path)

    grobid_parser = GrobidHeaderParser()

    with open(args.output_csv_path, "w", newline="") as csvfile:
        fields = [
            "SHA",
            "URL",
            "Model",
            "Title",
            "TitleScore",
            "Abstract",
            "AbstractScore",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for label in labels:
            # Known failing PDFs are excluded ...
            if label.id in [
                "396fb2b6ec96ff74e22ddd2484a9728257cccfbf",
                "3ef6e51baee01b4c90c188a964f2298b7c309b07",
                "4277d1ec41d88d595a0d80e4ab4146d8c2db2539",
                "564a73c07436e1bd75e31b54825d2ba8e4fb68b7",
            ]:
                continue

            save_prediction(writer, label, label, "Gold", 1.0, 1.0)

            item_pdf_path = pdf_path(label.id)

            grobid_pred = grobid_prediction(item_pdf_path, grobid_parser)
            title_scores["grobid"].append(score(label, grobid_pred, "title"))
            abstract_scores["grobid"].append(score(label, grobid_pred, "abstract"))
            save_prediction(
                writer,
                label,
                grobid_pred,
                "Grobid-0.7.0",
                title_scores["grobid"][-1],
                abstract_scores["grobid"][-1],
            )

            s2_pred = s2_prediction(label.id)
            title_scores["s2"].append(score(label, s2_pred, "title"))
            abstract_scores["s2"].append(score(label, s2_pred, "abstract"))
            save_prediction(
                writer,
                label,
                s2_pred,
                "S2-API",
                title_scores["s2"][-1],
                abstract_scores["s2"][-1],
            )

            layout_predictor = LayoutParserPredictor.from_pretrained(
                "lp://efficientdet/PubLayNet"
            )
            equation_layout_predictor = LayoutParserPredictor.from_pretrained(
                "lp://efficientdet/MFD"
            )

            for vila_model_name in [
                "ivila-block-layoutlm-finetuned-docbank",
                "ivila-block-layoutlm-finetuned-grotoap2",
                "hvila-block-layoutlm-finetuned-docbank",
                "hvila-row-layoutlm-finetuned-docbank",
                "hvila-block-layoutlm-finetuned-grotoap2",
                "hvila-row-layoutlm-finetuned-grotoap2",
            ]:
                vila_predictor, vila_extractor = _vila_models(vila_model_name)
                vila_doc = _load_pdf(item_pdf_path)

                vila_pred = vila_prediction(
                    label.id,
                    vila_doc,
                    vila_predictor=vila_predictor,
                    vila_extractor=vila_extractor,
                    layout_regions_predictor=layout_predictor,
                    equation_layout_regions_predictor=equation_layout_predictor,
                )

                title_scores[vila_model_name].append(score(label, vila_pred, "title"))
                abstract_scores[vila_model_name].append(
                    score(label, vila_pred, "abstract")
                )
                save_prediction(
                    writer,
                    label,
                    vila_pred,
                    vila_model_name,
                    title_scores[vila_model_name][-1],
                    abstract_scores[vila_model_name][-1],
                )

    for category, scores in {
        "TITLE": title_scores,
        "ABSTRACT": abstract_scores,
    }.items():
        print("-------- {} --------".format(category))

        for key in sorted(list(scores.keys())):
            data = scores[key]

            print(
                "{}---\nN: {}; Mean: {}; Std: {}".format(
                    key,
                    len(scores[key]),
                    mean(scores[key]),
                    stdev(scores[key]),
                )
            )
