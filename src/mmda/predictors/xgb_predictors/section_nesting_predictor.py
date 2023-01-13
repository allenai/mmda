"""

SectionNestingPredictor -- Use token-level predictions for "Section" to predict the
    parent-child relationships between sections.

Adapted from https://github.com/rauthur/section-annotations-gold

@rauthur

"""

import json
import logging
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import SpanGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.names import PagesField, SectionsField


@dataclass
class Example:
    parent_id: int
    parent_text: str
    parent_is_root: bool
    child_id: int
    child_text: str
    parent_no_font_size: bool
    child_no_font_size: bool
    is_one_size_larger_font: bool
    same_font: bool
    parent_bold_font: bool
    child_bold_font: bool
    normalized_page_distance: float
    on_same_page: bool
    relative_y_pos: int
    relative_x_pos: int
    abs_x_diff_pos: float
    abs_y_diff_pos: float
    parent_has_num_prefix: bool
    child_has_num_prefix: bool
    child_num_prefix_is_top_level: bool
    parent_prefix_is_implied_parent_of_child_prefix: bool
    child_text_starts_with_something_ending_with_a_period: bool
    child_is_top_level_keyword: bool
    child_is_all_caps: bool
    child_starts_with_upper_letter_prefix: bool
    parent_text_starts_with_something_ending_with_a_period: bool
    parent_is_top_level_keyword: bool
    parent_is_all_caps: bool
    parent_starts_with_upper_letter_prefix: bool


@dataclass
class _FontInfo:
    size: float
    name: str


class PdfStats:
    sections: Sequence[SpanGroup]
    section_index: Dict[int, SpanGroup]
    fontinfo_index: Dict[int, _FontInfo]

    def __init__(self, sections) -> None:
        self.sections = sections

        self._build_section_index()
        self._build_fontinfo_index()

    def section(self, id_: int) -> SpanGroup:
        return self.section_index[id_]

    def section_fontsize(self, id_: int) -> float:
        return self.fontinfo_index[id_].size

    def section_fontname(self, id_: int) -> str:
        return self.fontinfo_index[id_].name

    def _build_section_index(self):
        self.section_index = {s.id: s for s in self.sections}

    def _build_fontinfo_index(self):
        self.fontinfo_index = {}

        for section in self.sections:
            assert section.id is not None, "Sections must have an ID"

            self.fontinfo_index[section.id] = _FontInfo(
                size=self._round_size_with_default(section),
                name=self._fontname_with_default(section),
            )

    @cached_property
    def unique_fontsizes(self) -> List[float]:
        sizes = {self._round_size_with_default(s) for s in self.sections}
        return sorted([s for s in sizes if s > 0])

    def _round_size_with_default(self, section: SpanGroup, default=-1) -> float:
        return round(section.metadata.get("size", default), 4)

    def _fontname_with_default(self, section: SpanGroup, default="[NONE]") -> str:
        return section.metadata.get("fontname", default)


NUM_PREFIX_REGEX = "^([0-9\.]+)"


def num_prefix(s: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(NUM_PREFIX_REGEX, s)

    if m is None:
        return None, None

    s = m.group(0)
    if s.endswith("."):
        s = s[:-1]

    if "." in s:
        p = ".".join(s.split(".")[:-1])
    else:
        p = None

    return s, p


def child_text_starts_with_something_ending_with_a_period(
    s: SpanGroup,
) -> bool:
    if s.text is None:
        return False

    text = s.text.strip()

    if "." not in text:
        return False

    parts = text.split(" ")

    # There must be at least 2 words
    if len(parts) <= 1:
        return False

    # Ensure there is 1 occurence of '.' at the end of the word
    if not parts[0].endswith("."):
        return False
    if sum([1 if c == "." else 0 for c in parts[0]]) != 1:
        return False

    # Numbering should not be too long
    if len(parts[0]) > 3:
        return False

    return True


_TOP_LEVEL_KEYWORDS = [
    "abstract",
    "introduction",
    "conclusions",
    "references",
    "acknowledgements",
    "methods",
    "discussion",
    "keywords",
    "appendix",
]


def child_is_top_level_keyword(s: SpanGroup) -> bool:
    if s.text is None:
        return False

    text = s.text.strip().lower()

    # Trim any trailing punctuation like "Acknowledgements."
    if text.endswith(".") or text.endswith(":"):
        text = text[:-1]

    if len(text) == 0:
        return False

    # Single-word by may have some prefix like "I. Introduction"
    if len(text.split(" ")) > 2:
        return False

    for kw in _TOP_LEVEL_KEYWORDS:
        if text.endswith(kw):
            return True

    return False


def child_is_all_caps(s: SpanGroup) -> bool:
    if s.text is None:
        return False

    text = s.text.strip()

    if len(text) == 0:
        return False

    return text.upper() == text


def child_starts_with_upper_letter_prefix(s: SpanGroup) -> bool:
    if s.text is None:
        return False

    text = s.text.strip()

    if len(text) == 0:
        return False

    parts = text.split(" ")

    if len(parts) <= 1:
        return False
    if len(parts[0]) != 1:
        return False

    return parts[0] >= "A" and parts[0] <= "Z"


def span_group_page(span_group: SpanGroup) -> int:
    if len(span_group.spans) == 0:
        return -1

    return span_group.spans[0].box.page


SPAN_GROUP_ROOT = SpanGroup(spans=[], id=-1)


def make_example(
    pdf_stats: PdfStats,
    a: SpanGroup,
    b: SpanGroup,
    num_pages: int,
) -> Example:
    parent_is_root = a.id == -1

    a_font_size = -1 if parent_is_root else pdf_stats.section_fontsize(a.id)
    if a_font_size == -1:
        parent_no_font_size = True
    else:
        parent_no_font_size = False
    b_font_size = pdf_stats.section_fontsize(b.id)
    if b_font_size == -1:
        child_no_font_size = True
    else:
        child_no_font_size = False

    if (not parent_no_font_size) and (not child_no_font_size):
        a_font_index = pdf_stats.unique_fontsizes.index(a_font_size)
        b_font_index = pdf_stats.unique_fontsizes.index(b_font_size)
        assert a_font_index >= 0
        assert b_font_index >= 0
        is_one_size_larger_font = True if a_font_index - b_font_index == 1 else False
    else:
        is_one_size_larger_font = False

    parent_prefix, _ = num_prefix(a.text)
    child_prefix, implied_parent = num_prefix(b.text)

    b_box = Box.small_boxes_to_big_box(boxes=[s.box for s in b.spans])

    if parent_is_root:
        a_box = Box(l=0, t=0, w=0, h=0, page=0)

        relative_x_pos = -1
        relative_y_pos = -1

        a_font_name = "[ROOT]"
        b_font_name = pdf_stats.section_fontname(b.id)
    else:
        a_box = Box.small_boxes_to_big_box(boxes=[s.box for s in a.spans])

        relative_x_pos = 0
        if a_box.l < b_box.l:
            relative_x_pos = -1
        elif b_box.l < a_box.l:
            relative_x_pos = 1

        relative_y_pos = 0
        if a_box.t < b_box.t:
            relative_y_pos = -1
        elif b_box.t < a_box.t:
            relative_y_pos = 1

        a_font_name = pdf_stats.section_fontname(a.id)
        b_font_name = pdf_stats.section_fontname(b.id)

    a_page = span_group_page(a)
    b_page = span_group_page(b)

    return Example(
        parent_id=a.id,
        parent_text=a.text,
        parent_is_root=parent_is_root,
        child_id=b.id,
        child_text=b.text,
        parent_no_font_size=parent_no_font_size,
        child_no_font_size=child_no_font_size,
        same_font=a_font_name == b_font_name,
        parent_bold_font="bold" in a_font_name.lower(),
        child_bold_font="bold" in b_font_name.lower(),
        is_one_size_larger_font=is_one_size_larger_font,
        normalized_page_distance=(b_page - a_page) / num_pages,
        on_same_page=a_page == b_page,
        relative_x_pos=relative_x_pos,
        relative_y_pos=relative_y_pos,
        abs_x_diff_pos=abs(a_box.l - b_box.l),
        abs_y_diff_pos=abs(a_box.t - b_box.t),
        parent_has_num_prefix=not parent_prefix is None,
        child_has_num_prefix=not child_prefix is None,
        child_num_prefix_is_top_level=(
            child_prefix is not None and implied_parent is None
        ),
        parent_prefix_is_implied_parent_of_child_prefix=(
            parent_prefix is not None and parent_prefix == implied_parent
        ),
        child_text_starts_with_something_ending_with_a_period=child_text_starts_with_something_ending_with_a_period(
            b
        ),
        child_is_top_level_keyword=child_is_top_level_keyword(b),
        child_is_all_caps=child_is_all_caps(b),
        child_starts_with_upper_letter_prefix=child_starts_with_upper_letter_prefix(b),
        parent_text_starts_with_something_ending_with_a_period=child_text_starts_with_something_ending_with_a_period(
            a
        ),
        parent_is_top_level_keyword=child_is_top_level_keyword(a),
        parent_is_all_caps=child_is_all_caps(a),
        parent_starts_with_upper_letter_prefix=child_starts_with_upper_letter_prefix(a),
    )


@dataclass
class SectionNode:
    prev: Optional["SectionNode"]
    next: Optional["SectionNode"]
    section: SpanGroup


class SectionIndex:
    index: Dict[int, SectionNode]

    def __init__(self) -> None:
        self.index = OrderedDict()
        self.index[-1] = SectionNode(
            prev=None,
            next=None,
            section=SPAN_GROUP_ROOT,
        )

    def add(self, section: SpanGroup, parent_id: int):
        if parent_id not in self.index:
            raise ValueError("Cannot find parent!")

        parent = self.index[parent_id]

        curr = parent.next
        while curr is not None:
            currnext = curr.next
            del self.index[curr.section.id]
            del curr
            curr = currnext

        node = SectionNode(prev=parent, next=None, section=section)
        parent.next = node
        self.index[section.id] = node

    def __str__(self) -> str:
        curr = self.index[-1]
        nodes = []

        while curr is not None:
            nodes.append(f"[{curr.section.id}] {curr.section.text}")
            curr = curr.next

        return " -> ".join(nodes)


def bf(b: bool):
    return 1.0 if b else 0.0


def convert_example(x: Example):
    return [
        bf(x.is_one_size_larger_font),
        bf(x.same_font),
        bf(x.parent_no_font_size),
        bf(x.child_no_font_size),
        bf(x.parent_bold_font),
        bf(x.child_bold_font),
        x.normalized_page_distance,
        bf(x.on_same_page),
        x.relative_y_pos,
        x.relative_x_pos,
        x.abs_x_diff_pos,
        x.abs_y_diff_pos,
        bf(x.parent_has_num_prefix),
        bf(x.child_has_num_prefix),
        bf(x.child_num_prefix_is_top_level),
        bf(x.parent_prefix_is_implied_parent_of_child_prefix),
        bf(x.parent_is_root),
        bf(x.child_text_starts_with_something_ending_with_a_period),
        bf(x.child_is_top_level_keyword),
        bf(x.child_is_all_caps),
        bf(x.child_starts_with_upper_letter_prefix),
        bf(x.parent_text_starts_with_something_ending_with_a_period),
        bf(x.parent_is_top_level_keyword),
        bf(x.parent_is_all_caps),
        bf(x.parent_starts_with_upper_letter_prefix),
    ]


class SectionNestingPredictor(BasePredictor):
    REQUIRED_BACKENDS = None
    REQUIRED_DOCUMENT_FIELDS = [SectionsField, PagesField]

    def __init__(self, model_file: str) -> None:
        super().__init__()

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_file)

    def predict(self, document: Document) -> List[SpanGroup]:
        sections = document.sections

        if len(sections) == 0:
            return []

        index = SectionIndex()
        pdf_stats = PdfStats(sections)

        results = []

        for section in sections:
            xs = []

            # FIXME: Debugging only?
            parent_texts = []
            parent_ids = []

            for node in index.index.values():
                x = make_example(pdf_stats, node.section, section, len(document.pages))
                xs.append(convert_example(x))

                parent_texts.append(node.section.text)
                parent_ids.append(node.section.id)

            logging.debug("SECTION: %s [%i]", section.text, section.id)
            logging.debug("CANDIDATES: %s", json.dumps(parent_texts))

            pos_probs = self.model.predict_proba(xs)[:, 1]
            pos_probs = pos_probs / sum(pos_probs)
            pos_index = np.argmax(pos_probs)

            logging.debug(json.dumps([float(round(p, 4)) for p in pos_probs]))
            logging.debug(f"Picked {parent_texts[pos_index]}!")

            parent_id = parent_ids[pos_index]

            # Maintain the text from VILA for each span group
            metadata = deepcopy(section.metadata)
            metadata.parent_id = parent_id

            results.append(
                SpanGroup(
                    spans=deepcopy(section.spans),
                    box_group=deepcopy(section.box_group),
                    id=deepcopy(section.id),  # Ensure some ID is created
                    doc=None,  # Allows calling doc.annotate(...)
                    metadata=metadata,
                )
            )

            index.add(section, parent_ids[pos_index])

        return results
