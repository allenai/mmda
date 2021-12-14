"""

@rauthur

"""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from intervaltree import IntervalTree
from mmda.types.box import Box
from mmda.types.image import PILImage
from mmda.types.nouveau.user_data import UserDataMixin
from mmda.utils.tools import allocate_overlapping_tokens_for_box, merge_neighbor_spans


class Document(UserDataMixin):
    symbols: str
    indexers: Dict[str, "IntervalTreeIndexer"]
    images: Optional[Iterable[PILImage]]

    def __init__(self, symbols: str):
        super().__init__(after_set_callback=self._set_callback)

        self.symbols = symbols
        self.indexers = defaultdict(IntervalTreeIndexer)
        self.box_group_indexers = defaultdict(IntervalTreeIndexer)

    def attach_images(self, images: Iterable[PILImage]) -> None:
        """Attach page images to this document. Overwrites any that are present.

        Args:
            images (Iterable[PILImage]): Iterable of individual pages in order
        """
        self.images = images

    def _set_callback(self, name: str, values: Iterable[Any]) -> None:
        """Overrides method in UserDataMixin to attach document to all span/box groups.
        This method is also used to index annotations.

        Args:
            value (Any): A span or box group (DocAttachable)
        """
        for value in values:
            if isinstance(value, DocAttachable):
                value.attach_doc(self)

        if all([isinstance(value, SpanGroup) for value in values]):
            self._index_span_groups(name, values)

        if all([isinstance(value, BoxGroup) for value in values]):
            self._index_box_groups(name, values)

    def _index_span_groups(self, name: str, span_groups: Iterable["SpanGroup"]):
        span_group_indexer = self.indexers[name]

        if name == "bibliography":
            import pdb

            pdb.set_trace()

        # for each span in span_group
        #   (a) check if any conflicts (requires disjointedness)
        #   (b) then add span group to index at this span location
        for span_group in span_groups:
            for span in span_group.spans:
                matched_span_group = span_group_indexer[span.start : span.end]

                if matched_span_group:
                    raise ValueError(
                        f"Detected overlap with existing SpanGroup {matched_span_group} when attempting index {span_group}"
                    )

                span_group_indexer[span.start : span.end] = span_group

    def _index_box_groups(self, name: str, box_groups: Iterable["BoxGroup"]):
        all_page_tokens = dict()
        derived_span_groups: List[SpanGroup] = []

        for box_group in box_groups:
            all_token_spans_with_box_group = []

            for box in box_group.boxes:

                # Caching the page tokens to avoid duplicated search
                if box.page not in all_page_tokens:
                    all_page_tokens[box.page] = list(
                        itertools.chain.from_iterable(
                            span_group.spans
                            for span_group in self.pages[box.page].tokens
                        )
                    )

                # Each page token is assigned just once
                cur_page_tokens = all_page_tokens[box.page]

                # Find all the tokens within the box
                tokens_in_box, remaining_tokens = allocate_overlapping_tokens_for_box(
                    token_spans=cur_page_tokens, box=box
                )
                all_page_tokens[box.page] = remaining_tokens

                all_token_spans_with_box_group.extend(tokens_in_box)

            derived_span_group = SpanGroup(
                spans=merge_neighbor_spans(
                    spans=all_token_spans_with_box_group, distance=1
                ),
            )
            derived_span_group.box_group = box_group
            derived_span_groups.append(derived_span_group)

        derived_span_groups = sorted(
            derived_span_groups, key=lambda span_group: span_group.start
        )

        # Set ID for each span group by ordering
        for idx, derived_span_group in enumerate(derived_span_groups):
            derived_span_group.id = idx

        self._index_span_groups(name, derived_span_groups)

    @property
    def tokens(self) -> Iterable["SpanGroup"]:
        return self._.tokens

    @property
    def pages(self) -> Iterable["SpanGroup"]:
        return self._.pages

    @pages.setter
    def pages(self, value: Iterable["SpanGroup"]):
        self._.pages = value

    @property
    def rows(self) -> Iterable["SpanGroup"]:
        return self._.rows

    @property
    def images(self) -> Iterable[PILImage]:
        return self._.images

    @images.setter
    def images(self, images: Iterable[PILImage]):
        self._.images = images

    def symbols_for(self, span_group: "SpanGroup") -> Iterable[str]:
        return [self.symbols[s.start : s.end] for s in span_group.spans]

    def find_overlapping_span_groups(
        self, field_name: str, query: "SpanGroup"
    ) -> Iterable["SpanGroup"]:
        if not isinstance(query, SpanGroup):
            raise NotImplementedError(
                f"Currently only supports query of type SpanGroup"
            )

        if not field_name in self.indexers:
            raise KeyError(f"{field_name} was never added to indexed SpanGroups")

        return self.indexers[field_name].find(query=query)


class DocAttachable:
    doc: Optional[Document] = None

    def attach_doc(self, doc: Document):
        # FIXME: This check cannot happen now due to wrapping doc into ResearchArticle
        # if not self.doc:
        self.doc = doc
        # else:
        # raise AttributeError(f"This annotation already has an attached document")

    def attach_doc_callback(self, name: str, value: Any) -> None:
        """Attach document on this attachable to any child annotations.

        Args:
            value (Any): A span or box group (DocAttachable)
        """
        if self.doc and isinstance(value, DocAttachable):
            value.attach_doc(self)


@dataclass
class Span:
    start: int
    end: int
    box: Box

    @property
    def page(self):
        return self.box.page


@dataclass
class BoxGroup(UserDataMixin, DocAttachable):
    boxes: Iterable[Box]

    def __init__(self, boxes: Iterable[Box]):
        super().__init__(after_set_callback=self.attach_doc_callback)
        self.boxes = boxes

    @property
    def id(self) -> int:
        return self._.id

    @id.setter
    def id(self, value: int):
        self._.id = value


@dataclass
class SpanGroup(UserDataMixin, DocAttachable):
    spans: Iterable[Span]

    def __init__(self, spans: Iterable[Span]):
        super().__init__(after_set_callback=self.attach_doc_callback)
        self.spans = spans

    @property
    def id(self) -> int:
        return self._.id

    @id.setter
    def id(self, value: int):
        self._.id = value

    @property
    def symbols(self) -> Iterable[str]:
        return [self.doc.symbols[span.start : span.end] for span in self.spans]

    @property
    def tokens(self) -> Iterable["SpanGroup"]:
        return self.doc.find_overlapping_span_groups("tokens", self)

    @property
    def rows(self) -> Iterable["SpanGroup"]:
        return self.doc.find_overlapping_span_groups("rows", self)

    @property
    def blocks(self) -> Iterable["SpanGroup"]:
        return self.doc.find_overlapping_span_groups("blocks", self)

    @property
    def box_group(self) -> "BoxGroup":
        return self._.box_group

    @box_group.setter
    def box_group(self, value: "BoxGroup"):
        self._.box_group = value

    def __getitem__(self, key: int):
        return self.spans[key]

    @property
    def start(self) -> int:
        return (
            min([span.start for span in self.spans])
            if len(self.spans) > 0
            else float("-inf")
        )

    @property
    def end(self) -> int:
        return (
            max([span.end for span in self.spans])
            if len(self.spans) > 0
            else float("inf")
        )

    def __lt__(self, other: "SpanGroup"):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start


@dataclass
class IntervalTreeIndexer:

    _index: IntervalTree

    def __init__(self) -> None:
        self._index = IntervalTree()

    def find(self, query: SpanGroup) -> Iterable[SpanGroup]:
        if not isinstance(query, SpanGroup):
            raise ValueError(
                f"IntervalTreeSpanGroupIndexer only works with `query` that is SpanGroup type"
            )

        all_matched_span_groups = []

        for span in query.spans:
            for matched_span_group in self._index[span.start : span.end]:
                if matched_span_group.data not in all_matched_span_groups:
                    all_matched_span_groups.append(matched_span_group.data)

        # retrieval can be out of order, so sort
        return sorted(all_matched_span_groups)

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key, value):
        self._index[key] = value
