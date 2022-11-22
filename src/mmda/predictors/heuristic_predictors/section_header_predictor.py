"""

SectionHeaderPredictor -- Use PDF outline metadata to predict section headers and
    levels. This predictor is entirely heuristic and only applies to PDFs that have ToC
    information in the sidebar. See SectionNestingPredictor for a related class that
    operates over token predictions to yield an outline of sections.

Adapted from https://github.com/rauthur/pdf-outlines-extraction

@rauthur

"""


from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, List

from layoutparser.tools.shape_operations import (
    generalized_connected_component_analysis_1d,
)

from mmda.eval.metrics import levenshtein
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import Span, SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.metadata import Metadata
from mmda.types.names import PagesField, TokensField
from mmda.utils.outline_metadata import Outline, OutlineItem


@dataclass
class _BoxWithText:
    span: Span
    box: Box
    text: str


def _doc_has_no_outlines(doc: Document) -> bool:
    return "outline" not in doc.metadata or len(doc.metadata.outline) == 0


def _parse_outline_metadata(doc: Document) -> Outline:
    return Outline.from_metadata_dict(doc.metadata)


def _outlines_to_page_index(outline: Outline) -> Dict[int, List[OutlineItem]]:
    results = defaultdict(list)

    for item in outline.items:
        results[item.page].append(item)

    return results


def _guess_box_dimensions(spans: List[Span], index: int, outline: OutlineItem) -> Box:
    """Use this method to expand a pointer to a location in a PDF to a guesstimate box
    that represents a full target location for a Box after clicking a ToC entry. In
    other words, this box is roughly on the PDF page where PDF software would scroll to
    if clicking the sidebar entry. It makes use of the surrounding token boxes to try to
    capture a reasonable area.

    Args:
        page (SpanGroup): The page object from a PDF parser
        index (int): The page index from 0
        outline (OutlineMetadata): Rehydrated OutlineMetadata object from querier

    Returns:
        Box: A box that approximately points to the ToC location of interest
    """
    box = Box(
        l=outline.l,
        t=outline.t,
        w=0.25 if outline.l > 0 else 1.0,  # 25% to the right if left else 100% of page
        h=0.10,  # 10% to the bottom
        page=index,
    )

    boxes: List[Box] = [s.box for s in spans]

    # Consider tokens below the top of the box
    # Consider tokens within left/right boundaries
    # Take the closest top here and use as bottom
    borderline = min(
        *[
            b.t
            for b in boxes
            if b.t + 0.01 > box.t and b.l + 0.01 > box.l and b.l < box.l + box.w
        ]
    )

    # Add a small fudge factor and shrink the top since we have a better box
    box.t = borderline - 0.005
    box.h = 0.01

    return box


def _x_in_thresh(a: Box, b: Box, thresh) -> bool:
    if a.is_overlap(b):
        return True

    items = sorted([a, b], key=lambda x: x.l)
    return items[1].l - (items[0].l + items[0].w) <= thresh


def _y_in_thresh(a: Box, b: Box, thresh) -> bool:
    if a.is_overlap(b):
        return True

    items = sorted([a, b], key=lambda x: x.t)
    return items[1].t - (items[0].t + items[0].h) <= thresh


def _edge_threshold(
    a: _BoxWithText, b: _BoxWithText, x_thresh: float, y_thresh: float
) -> int:
    return int(
        _x_in_thresh(a.box, b.box, x_thresh) and _y_in_thresh(a.box, b.box, y_thresh)
    )


_lscore = partial(levenshtein, case_sensitive=False, strip_spaces=True, normalize=True)


def _find_best_candidate(
    candidates: List[List[_BoxWithText]], outline: OutlineItem
) -> List[_BoxWithText]:
    best_candidate = candidates[0]
    best_text = "".join([x.text for x in best_candidate])
    best_score = _lscore(outline.title, best_text)

    if len(candidates) == 1:
        return best_candidate

    for other_candidate in candidates[1:]:
        other_text = "".join(x.text for x in other_candidate)
        other_score = _lscore(outline.title, other_text)

        if other_score < best_score:
            best_candidate = other_candidate
            best_score = other_score
            best_text = other_text

    return best_candidate


MAGIC_TOKEN = "[^^ SHP_TARGET ^^]"


class SectionHeaderPredictor(BasePredictor):

    REQUIRED_BACKENDS = None
    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField]

    _x_threshold: float
    _y_threshold: float

    def __init__(
        self,
        _x_threshold: float = 0.015,
        _y_threshold: float = 0.005,
    ) -> None:
        self._x_threshold = _x_threshold
        self._y_threshold = _y_threshold

    def predict(self, document: Document) -> List[SpanGroup]:
        """Get section headers in a Document as a list of SpanGroup.

        Args:
            doc (Document): The document to process

        Returns:
            list[SpanGroup]: SpanGroups that appear to be headers based on outline
                metadata in the PDF (i.e., ToC or sidebar headers).
        """
        if _doc_has_no_outlines(document):
            return []

        self._doc_field_checker(document)

        outlines = _parse_outline_metadata(document)
        page_to_outlines = _outlines_to_page_index(outlines)
        predictions: List[SpanGroup] = []

        for i, page in enumerate(document.pages):
            tokens: List[SpanGroup] = page.tokens
            spans: List[Span] = [s for t in tokens for s in t.spans]

            for outline in page_to_outlines[i]:

                box: Box = _guess_box_dimensions(spans, i, outline)

                text_boxes: List[_BoxWithText] = [
                    _BoxWithText(
                        span=span,
                        box=span.box,
                        text=document.symbols[span.start : span.end],
                    )
                    for span in spans
                    if (
                        span.box.t + 0.005 > box.t
                        and span.box.t + span.box.h < box.t + 0.25
                    )
                ]
                text_boxes.append(_BoxWithText(span=None, box=box, text=MAGIC_TOKEN))

                components: List[
                    List[_BoxWithText]
                ] = generalized_connected_component_analysis_1d(
                    text_boxes,
                    partial(
                        _edge_threshold,
                        x_thresh=self._x_threshold,
                        y_thresh=self._y_threshold,
                    ),
                )

                for component in components:
                    component_texts = [x.text for x in component]

                    if MAGIC_TOKEN not in component_texts:
                        continue

                    magic_token = component[-1]

                    # Filter out chars that start above the target token
                    filtered = [x for x in component if x.box.t >= magic_token.box.t]

                    # Prune target since we no longer need this
                    filtered = [x for x in filtered if x.text != MAGIC_TOKEN]

                    # Try to find the best of a few different filters
                    # Initial candidate is simply all linked tokens
                    # Additional candidates could use things like font name and size
                    candidates = [
                        [x for x in filtered],
                    ]

                    # Add candidate that drops tokens to the left if non-zero left
                    if magic_token.box.l > 0:
                        candidates.append(
                            [x for x in filtered if x.box.l + 0.01 > magic_token.box.l]
                        )

                    # If all tokens are filtered out then we did not find a match
                    if len(candidates) == 0:
                        continue

                    best_candidate = _find_best_candidate(candidates, outline)
                    metadata = Metadata(level=outline.level, title=outline.title)
                    predictions.append(
                        SpanGroup(
                            spans=[x.span for x in best_candidate], metadata=metadata
                        )
                    )

        return predictions
