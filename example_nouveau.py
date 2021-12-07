from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.nouveau_vila_predictor import NouveauHVilaPredictor
from mmda.predictors.tesseract_predictors import TesseractBlockPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.nouveau.protocol import BoxPredictor, Parser, Rasterizer, SpanPredictor
from mmda.types.nouveau.scientific import ResearchArticle, TitleExtractor

PDF = "resources/maml.pdf"

parser: Parser = PDFPlumberParser()
rasterizer: Rasterizer = PDF2ImageRasterizer()

# First we'll parse the document and attach images
doc: ResearchArticle = parser.parse(PDF)
doc.attach_images(rasterizer.convert(PDF, dpi=72))

# Now we need model-specific details like text blocks
block_predictor: BoxPredictor = TesseractBlockPredictor()
doc.blocks = block_predictor.predict(doc)

# Which can inform token-level predictions
token_predictor: SpanPredictor = NouveauHVilaPredictor.from_pretrained(
    "allenai/hvila-row-layoutlm-finetuned-grotoap2",
    agg_level="block",
    added_special_sepration_token="[BLK]",
    group_bbox_agg="first",
)
doc.tokens = token_predictor.predict(doc)

import pdb

pdb.set_trace()


# And now we can organize information for an end-user
title_extractor: TitleExtractor = None  # FIXME
doc.title = title_extractor.extract(doc)


for bib in doc.bibliography:
    print(bib.text)
    print(bib.title)

    for author in bib.authors:
        print(author.text)

        for token in author.tokens:
            print(token)
