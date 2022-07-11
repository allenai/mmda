#!/usr/bin/env python3
import pathlib
import sys

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.hf_predictors.mention_predictor import MentionPredictor

pdf_file = pathlib.Path(sys.argv[1]).resolve()
print(f"reading pdf from from {pdf_file}\n")

pdf_plumber = PDFPlumberParser(split_at_punctuation=True)
doc = pdf_plumber.parse(str(pdf_file))

# download weights to ${git-root}/temp/weights/
artifacts_dir = pathlib.Path(__file__).parent.parent / "temp" / "weights"
mentions = MentionPredictor(str(artifacts_dir)).predict(doc)
doc.annotate(mentions=mentions)

for mention in doc.mentions:
    print(mention.symbols)
