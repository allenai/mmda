# MMDA is based on this commit https://github.com/allenai/mmda/pull/63/commits/c5b4a2ffe1450312af57b6acd31598a4f130635b
# 
# Usage
#
# python eval.py \
#     --use_symbol_scraper \
#     --symbol_scraper_bin /net/nfs2.s2-research/shannons/gitsource/SymbolScraper/bin/sscraper
# 
# It will load the PDF metadata in `eval_data_large.jsonl` and assume all the PDFs have been downloaded
# to the `pdfs/<sha>.pdf`. 

import json
import pandas as pd
from tqdm import tqdm
import argparse

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.rasterizers.rasterizer import Rasterizer
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import (
    IVILAPredictor,
    HVILAPredictor,
    SimpleVILAPredictor,
)


LABEL_MAP = {
    "docbank": {
        "0": "paragraph",
        "1": "title",# <--
        "2": "equation",
        "3": "reference",
        "4": "section",
        "5": "list",
        "6": "table",
        "7": "caption",
        "8": "author",
        "9": "abstract",# <--
        "10": "footer",
        "11": "date",
        "12": "figure",
    },
    "s2-hard": {
        "0": "Title",# <--
        "1": "Author",
        "2": "Abstract",# <--
        "3": "Keywords",
        "4": "Section",
        "5": "Paragraph",
        "6": "List",
        "7": "Bibliography",
        "8": "Equation",
        "9": "Algorithm",
        "10": "Figure",
        "11": "Table",
        "12": "Caption",
        "13": "Header",
        "14": "Footer",
        "15": "Footnote",
    },
    "s2-vl": {
        "0": "Title",# <--
        "1": "Author",
        "2": "Abstract",# <--
        "3": "Keywords",
        "4": "Section",
        "5": "Paragraph",
        "6": "List",
        "7": "Bibliography",
        "8": "Equation",
        "9": "Algorithm",
        "10": "Figure",
        "11": "Table",
        "12": "Caption",
        "13": "Header",
        "14": "Footer",
        "15": "Footnote",
    },
    "grotoap2": {
        "0": "BIB_INFO",
        "1": "REFERENCES",
        "2": "UNKNOWN",
        "3": "BODY_CONTENT",
        "4": "PAGE_NUMBER",
        "5": "TABLE",
        "6": "ACKNOWLEDGMENT",
        "7": "FIGURE",
        "8": "CONFLICT_STATEMENT",
        "9": "AFFILIATION",
        "10": "DATES",
        "11": "TYPE",
        "12": "TITLE", # <--
        "13": "AUTHOR",
        "14": "ABSTRACT", # <--
        "15": "CORRESPONDENCE",
        "16": "EDITOR",
        "17": "COPYRIGHT",
        "18": "AUTHOR_TITLE",
        "19": "KEYWORDS",
        "20": "GLOSSARY",
        "21": "EQUATION",
    },
}

LABEL_MAP = {
    dataset_name: {int(idx): label.lower() for idx, label in label_map.items()}
    for dataset_name, label_map in LABEL_MAP.items()
}


def construct_pred_name(model_config):
    return "text_pred_" + "_".join(ele for ele in model_config if ele is not None)


def load_jsonl(path: str):
    """Loads a jsonl file and returns a list of Document objects

    Args:
        path (str): Path to the jsonl file

    Returns:
        List[Document]
    """
    with open(path, "r") as f:
        documents = [json.loads(line) for line in f]
    return documents


PDF_METADATA = load_jsonl("eval_data_large.jsonl")
TOP_N_PAGES = 2


def load_pdf(pdf_path):

    pdfplumber_parser = PDFPlumberParser()
    doc = pdfplumber_parser.parse(pdf_path)

    rasterizer = Rasterizer()
    doc.images = rasterizer.rasterize(pdf_path, dpi=72)

    doc.pages = doc.pages[:TOP_N_PAGES]
    doc.images = doc.images[:TOP_N_PAGES]
    # A hack
    # We assume the titles and abstract only appears within the first TOP_N_PAGES pages
    # Thus we drop the subsequent pages to speed up

    return doc


def load_pdf_symbol_scraper(pdf_path, symbol_scraper_bin):

    pdfplumber_parser = SymbolScraperParser(symbol_scraper_bin)
    doc = pdfplumber_parser.parse(pdf_path)

    rasterizer = Rasterizer()
    doc.images = rasterizer.rasterize(pdf_path, dpi=72)

    doc.pages = doc.pages[:TOP_N_PAGES]
    doc.images = doc.images[:TOP_N_PAGES]
    # A hack
    # We assume the titles and abstract only appears within the first TOP_N_PAGES pages
    # Thus we drop the subsequent pages to speed up
    return doc


def create_text_predictors():
    # training_datasets = ["s2-vl", "docbank", "s2-hard", "grotoap2"]
    training_datasets = ["s2-vl", "grotoap2"]
    model_families = ["layoutlm", "ivila", "hvila"]
    agg_levels = ["row", "block"]

    for training_dataset in training_datasets:
        for model_family in model_families:
            if model_family == "ivila":
                for agg_level in agg_levels:
                    model_path = f"models/{training_dataset}/layout_indicator-BLK-{agg_level}/microsoft-layoutlm-base-uncased/"
                    if training_dataset == "s2-hard":
                        model_path += "70/0/"
                        if agg_level == "row":
                            continue
                    elif training_dataset == "s2-vl":
                        model_path += "0/"

                    text_predictor = IVILAPredictor.from_pretrained(
                        model_path,
                        added_special_sepration_token="[BLK]",
                        agg_level=agg_level,
                    )
                    model_config = (training_dataset, model_family, agg_level)
                    yield text_predictor, model_config

            elif model_family == "hvila":
                for agg_level in agg_levels:
                    model_path = f"models/{training_dataset}/hierarchical_model-{agg_level}/weak-strong-layoutlm-average-first/"
                    if training_dataset == "s2-hard":
                        model_path += "70/0/"
                        if agg_level == "row":
                            continue
                    elif training_dataset == "s2-vl":
                        model_path += "0/"

                    text_predictor = HVILAPredictor.from_pretrained(
                        model_path, agg_level=agg_level, group_bbox_agg="first"
                    )
                    model_config = (training_dataset, model_family, agg_level)
                    yield text_predictor, model_config

            else:
                model_path = f"models/{training_dataset}/baseline-long-run/microsoft-layoutlm-base-uncased/"
                if training_dataset == "s2-hard":
                    model_path += "70/0/"
                elif training_dataset == "s2-vl":
                    model_path += "0/"

                text_predictor = SimpleVILAPredictor.from_pretrained(
                    model_path,
                    added_special_sepration_token="[BLK]",
                    agg_level="block",
                )
                model_config = (training_dataset, model_family, None)
                yield text_predictor, model_config


ALL_TEXT_PREDICTORS = list(create_text_predictors())


def load_text_predictor():
    """The create_text_predictors preloads all the text predictors.
    And the load_text_predictor is an actual generator that returns the predictor.
    """
    for text_predictor, model_config in ALL_TEXT_PREDICTORS:
        yield text_predictor, model_config


def parse_outputs(doc, pred_name, training_dataset):
    preds = [
        (LABEL_MAP[training_dataset][pred.type], " ".join(pred.symbols))
        for pred in getattr(doc, pred_name)
    ]
    return pd.DataFrame(preds, columns=["type", "text"])


def extract_title(df):
    return "\n".join(df[df["type"] == "title"]["text"].tolist())


def extract_abstract(df):
    return "\n".join(df[df["type"] == "abstract"]["text"].tolist())


if __name__ == "__main__":
    layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/PubLayNet"
    )
    equation_layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/MFD"
    )


    # load args for part 
    parser = argparse.ArgumentParser()
    

    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--total-parts", type=int, default=2)

    parser.add_argument("--use_symbol_scraper", action="store_true")
    parser.add_argument("--symbol_scraper_bin", type=str, default=None)

    args = parser.parse_args()

    BASE_PDF_PATH = "pdfs"

    all_predictions = []
    all_gt = []

    if args.part is None:
        pbar = tqdm(PDF_METADATA)
    else:
        total_parts = args.total_parts
        part_len = len(PDF_METADATA) // total_parts
        cur_part = args.part
        if cur_part +1 == total_parts:
            pbar = tqdm(PDF_METADATA[cur_part*part_len:])
        else:    
            pbar = tqdm(PDF_METADATA[cur_part*part_len:(cur_part+1)*part_len])

    for pdf_data in pbar:
        pbar.set_description(pdf_data["pdf_hash"])
        
        try:
            pdf_path = f"{BASE_PDF_PATH}/{pdf_data['pdf_hash']}.pdf"
            if args.use_symbol_scraper:
                doc = load_pdf_symbol_scraper(pdf_path, args.symbol_scraper_bin)
            else:
                doc = load_pdf(pdf_path)

            all_gt.append(
                (
                    pdf_data["pdf_hash"],
                    pdf_data["publisher_title"],
                    pdf_data["publisher_abstract"],
                )
            )

            # Obtaining Layout Predictions
            layout_regions = layout_predictor.predict(
                doc
            )  # Detect content regions like paragraphs
            equation_layout_regions = equation_layout_predictor.predict(
                doc
            )  # Detect equation regions
            doc.annotate(blocks=layout_regions + equation_layout_regions)

            # VILA Predictions
            for text_predictor, model_config in load_text_predictor():
                pred_name = construct_pred_name(model_config)
                doc.annotate(**{pred_name: text_predictor.predict(doc)})
                df = parse_outputs(doc, pred_name, model_config[0])
                cur_title = extract_title(df)
                cur_abstract = extract_abstract(df)
                all_predictions.append(
                    [
                        pdf_data["pdf_hash"],
                        *model_config,
                        cur_title,
                        cur_abstract,
                    ]
                )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Fail to process pdf {pdf_path} for the following error:\n{error_msg}")
            continue

    all_predictions = pd.DataFrame(
        all_predictions,
        columns=[
            "sha",
            "training_dataset",
            "model_family",
            "agg_level",
            "title",
            "abstract",
        ],
    )
    all_gt = pd.DataFrame(
        all_gt, columns=["sha", "publisher_title", "publisher_abstract"]
    )

    if args.part is None:
        all_predictions.to_csv("all_predictions.csv", index=None)
        all_gt.to_csv("all_gt.csv", index=None)
    else:
        all_predictions.to_csv(f"all_predictions.csv.part.{args.part}", index=None)
        all_gt.to_csv(f"all_gt.csv.part.{args.part}", index=None)
