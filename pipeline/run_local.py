from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.predictors.hf_predictors.vila_predictor import IVILAPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer

pdf_file = "/home/ubuntu/git/VILA/2306c568f2d3dfec6762ccb9fb16e63e173a.pdf"
print(f"reading pdf from from {pdf_file}")

pdf_plumber = PDFPlumberParser()
rasterizer = PDF2ImageRasterizer()

doc = pdf_plumber.parse(pdf_file)
doc.annotate_images(rasterizer.rasterize(pdf_file, dpi=72))
    
lp_predictor1 = LayoutParserPredictor.from_pretrained("lp://efficientdet/PubLayNet")
lp_predictor2 = LayoutParserPredictor.from_pretrained("lp://efficientdet/MFD")
blocks = lp_predictor1.predict(doc) + lp_predictor2.predict(doc)
doc.annotate(blocks=blocks)

vila_predictor = IVILAPredictor.from_pretrained(
    "/home/ubuntu/tmp/vila",
    added_special_sepration_token="[BLK]",
    agg_level="row"
)
doc.annotate(vila_spans=vila_predictor.predict(doc))

dictionary_word_predictor = DictionaryWordPredictor("/dev/null")
words = dictionary_word_predictor.predict(doc)
doc.annotate(words=words)

title = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == 0
)
abstract = " ".join(
    " ".join(w.text for w in sg.words)
    for sg in doc.vila_spans if sg.type == 2
)

print(f"Title    = '{title}'")
print(f"Abstract = '{abstract}'")
