from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.nouveau_vila_predictor import NouveauHVilaPredictor
from mmda.predictors.tesseract_predictors import TesseractBlockPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.nouveau.base import SpanGroup
from mmda.types.nouveau.protocol import Parser, Rasterizer
from mmda.types.nouveau.scientific import (
    ResearchArticle,
    TokenGroupBibliographyExtractor,
    TokenGroupTitleExtractor,
)


def color(color: str, node: SpanGroup):
    print(f"Coloring {node} with {color}!")


def color_red(node: SpanGroup):
    color("RED", node)


def color_blue(node: SpanGroup):
    color("BLUE", node)


PDF = "resources/maml.pdf"

parser: Parser = PDFPlumberParser()
rasterizer: Rasterizer = PDF2ImageRasterizer()

# First we'll parse the document and attach images
doc = ResearchArticle.from_document(parser.parse(PDF))
doc.attach_images(rasterizer.convert(PDF, dpi=150))

# Now we need model-specific details like text blocks
block_predictor = TesseractBlockPredictor()
doc.blocks = block_predictor.predict(doc)

# Overwrite images b/c of what HVILA expects
doc.attach_images(rasterizer.convert(PDF, dpi=72))

# Which can inform token-level predictions
token_predictor = NouveauHVilaPredictor.from_pretrained(
    "allenai/hvila-row-layoutlm-finetuned-grotoap2",
    agg_level="block",
    added_special_sepration_token="[BLK]",
    group_bbox_agg="first",
)
doc._.preds = token_predictor.predict(doc)

# And now we can organize information for an end-user
title_extractor = TokenGroupTitleExtractor()
doc.title = title_extractor.extract(doc)

bibliography_extractor = TokenGroupBibliographyExtractor()
doc.bibliography = bibliography_extractor.extract(doc)
import pdb

pdb.set_trace()


for bib in doc.bibliography:
    print(bib.text)
    print(bib.title)

    for author in bib.authors:
        print(author.text)

        color_red(author.first_name.tokens)
        color_blue(author.first_name.tokens)
