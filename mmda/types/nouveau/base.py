"""

@rauthur

"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from intervaltree import IntervalTree
from mmda.types.box import Box
from mmda.types.image import PILImage
from mmda.types.nouveau.user_data import UserDataMixin


class Document(UserDataMixin):
    symbols: str
    indexers: Dict[str, Any]
    images: Optional[Iterable[PILImage]]

    def __init__(self, symbols: str):
        super().__init__(after_set_callback=self._set_callback)

        self.symbols = symbols
        self.indexers = defaultdict(IntervalTreeSpanGroupIndexer)

    def attach_images(self, images: Iterable[PILImage]) -> None:
        """Attach page images to this document. Overwrites any that are present.

        Args:
            images (Iterable[PILImage]): Iterable of individual pages in order
        """
        self.images = images

    def _set_callback(self, name: str, value: Any) -> None:
        """Overrides method in UserDataMixin to attach document to all span/box groups.

        Args:
            value (Any): A span or box group (DocAttachable)
        """
        if isinstance(value, DocAttachable):
            value.attach_doc(self)

        if isinstance(value, SpanGroup):
            self._index_span_group(name, value)

    def _index_span_group(self, name: str, span_group: Iterable["SpanGroup"]) -> None:
        """Annotate the Document using a bunch of span groups.
        It will associate the annotations with the document symbols.
        """
        span_group_indexer = self.indexers[name]

        # for each span in span_group
        #   (a) check if any conflicts (requires disjointedness)
        #   (b) then add span group to index at this span location
        for span in span_group:
            # a) Check index if any conflicts (we require disjointness)
            matched_span_group = span_group_indexer[span.start : span.end]

            if matched_span_group:
                raise ValueError(
                    f"Detected overlap with existing SpanGroup {matched_span_group} when attempting index {span_group}"
                )

            # b) If no issues, add to index (for each span in span_group)
            span_group_indexer[span.start : span.end] = span_group

    def _index_box_group(self):
        # TODO: What we want is the ability to query with a BoxGroup
        # Receive back all SpanGroups that have spans that overlap this BoxGroup
        # Consider... Should ALL boxes in SpanGroup overlap? What to do with a SpanGroup
        # that is a mix of overlapping and non-overlapping boxes with query BoxGroup?
        pass

    @property
    def tokens(self) -> Iterable["SpanGroup"]:
        return self._.tokens

    @property
    def pages(self) -> Iterable["SpanGroup"]:
        return self._.pages

    # FIXME: This may be optional or moved to lower level class
    @property
    def rows(self) -> Iterable["SpanGroup"]:
        return self._.rows

    @property
    def images(self) -> Iterable[PILImage]:
        return self._.images

    @images.setter
    def images(self, images: Iterable[PILImage]):
        self._.images = images

    # FIXME: This isn't inclusive of BoxGroups like "blocks"
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


@dataclass
class SpanGroup(UserDataMixin, DocAttachable):
    spans: Iterable[Span]

    def __init__(self, spans: Iterable[Span]):
        super().__init__(after_set_callback=self.attach_doc_callback)
        self.spans = spans

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
    def blocks(self) -> Iterable["BoxGroup"]:
        return self.doc.find_overlapping_box_groups("blocks", self)

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
class IntervalTreeSpanGroupIndexer:

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
                if (
                    matched_span_group.data not in all_matched_span_groups
                ):  # Deduplicate
                    all_matched_span_groups.append(matched_span_group.data)

        # retrieval can be out of order, so sort
        return sorted(all_matched_span_groups)

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key, value):
        self._index[key] = value
