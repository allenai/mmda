from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional

from mmda.eval.metrics import box_overlap
from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.heuristic_predictors.grobid_citation_predictor import (
    get_title,
)
from mmda.predictors.hf_predictors.vila_predictor import HVILAPredictor
from mmda.predictors.tesseract_predictors import TesseractBlockPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import BoxGroup, Entity
from mmda.types.document import Document

PDF_PATH = "resources/maml.pdf"


def _clone_span_group(span_group: Entity):
    return Entity(
        spans=span_group.spans,
        id=span_group.id,
        text=span_group.text,
        type=span_group.type,
        box_group=span_group.box_group,
    )


@dataclass
class PageSpan:
    num: int
    start: int
    end: int


def _index_document_pages(document) -> List[PageSpan]:
    page_spans = []

    # Assume document.pages yields SpanGroups with only 1 Span
    for page in document.pages:
        assert len(page.spans) == 1

        span = page.spans[0]
        page_spans.append(PageSpan(num=span.box.page, start=span.start, end=span.end))

    return page_spans


def _find_page_num(span_group: Entity, page_spans: List[PageSpan]) -> int:
    s = min([span.start for span in span_group.spans])
    e = max([span.end for span in span_group.spans])

    for page_span in page_spans:
        if s >= page_span.start and e <= page_span.end:
            return page_span.num

    raise RuntimeError(f"Unable to find page for {span_group}!")


def _highest_overlap_block(
    token: Entity, blocks: Iterable[BoxGroup]
) -> Optional[BoxGroup]:
    assert len(token.spans) == 1
    token_box = token.spans[0].box

    found_block = None
    found_score = 0.0

    for curr_block in blocks:
        assert len(curr_block.boxes) == 1
        curr_box = curr_block.boxes[0]
        curr_score = box_overlap(token_box, curr_box)

        if curr_score > found_score:
            found_score = curr_score
            found_block = curr_block

    return found_block


def extract_bibliography_grotoap2(document: Document) -> Iterable[Entity]:
    """GROTOAP2 has type 1 for REFERENCES"""
    return [_clone_span_group(sg) for sg in document.preds if sg.type == 1]


parser = PDFPlumberParser()
document = parser.parse(PDF_PATH)
rasterizer = PDF2ImageRasterizer()

# Use larger DPI for better results (supposedly) with Tesseract
images = rasterizer.rasterize(PDF_PATH, dpi=150)
document.annotate_images(images)

block_predictor = TesseractBlockPredictor()
blocks = block_predictor.predict(document)
document.annotate(blocks=blocks)

# Use smaller DPI images for VILA
images = rasterizer.rasterize(PDF_PATH, dpi=72)
document.annotate_images(images, is_overwrite=True)

vila_predictor = HVILAPredictor.from_pretrained(
    "allenai/hvila-row-layoutlm-finetuned-grotoap2",
    agg_level="block",
    added_special_sepration_token="[BLK]",
    group_bbox_agg="first",
)
preds = vila_predictor.predict(document)
document.annotate(preds=preds)

biblio = extract_bibliography_grotoap2(document)
document.annotate(bibliography=biblio)


# TESTING
bib = document.bibliography[0]
page_num = _find_page_num(bib, _index_document_pages(document))
# FIXME: We have to reference blocks b/c they've been coerced to SpanGroup in Document
page_blocks = [b for b in blocks if b.boxes[0].page == page_num]
block_tokens = defaultdict(list)

import pdb

pdb.set_trace()

# Group bib tokes into a block
for token in bib.tokens:
    highest_overlap_block = _highest_overlap_block(token, page_blocks)

    if highest_overlap_block:
        block_tokens[highest_overlap_block.id].append(token)


# Print bibliography entries
for span_groups in block_tokens.values():
    reference = " ".join(["".join(sg.symbols) for sg in span_groups])
    reference = reference.replace("- ", "")

    # Use grobid to get the title information from reference text
    title = get_title(reference)

    print(reference)
    print(title)
    print()
    print()
