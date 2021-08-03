"""

Annotations are objects that are 'aware' of the Document

Collections of Annotations are how one constructs a new Iterable of Group-type objects within the Document

"""

from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field


from mmda.types.span import Span, SpanGroup
from mmda.types.box import Box, BoxGroup

from intervaltree import IntervalTree


@dataclass
class Annotation:
    """Annotation is intended for storing model predictions for a document."""

    doc: Optional["Document"] = field(default=False, init=False)
    # Specify an attribute with default value in the parent class
    # Ref: https://stackoverflow.com/a/58525728

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    @abstractmethod
    def annotate(self, doc: "Document"):
        """Annotate the object itself on a specific document.
        It will associate the annotations with the document symbols.
        """

    # TODO[kylel] - comment explaining
    def __getattr__(self, field: str):
        if field in self.doc.fields:
            return self.doc.find(self, field)
        else:
            return self.__getattribute__(field)


@dataclass
class DocSpanGroup(Annotation):
    span_group: SpanGroup
    text: Optional[str] = None
    type: Optional[str] = None
    box_group: Optional[BoxGroup] = None

    def annotate(self, doc: "Document", field_name: str) -> None:
        """Annotate the object itself on a specific document.
        It will associate the annotations with the document symbols.
        """
        self.doc = doc
        doc_field_indexer = doc._indexers[field_name]

        for span in SpanGroup:
            # constraint - all spans disjoint
            existing = doc_field_indexer[span.page_id][span.start:span.end]
            if existing:
                raise ValueError(f'Existing {existing} when attempting index {span}')
            # add to index
            doc_field_indexer[span.page_id][span.start:span.end] = self

    def to_json(self) -> Dict:
        return dict(
            _type="DocSpanGroup",  # Used for differenting between DocSpan and DocBox when loading the json
            span_group=self.span_group.to_json(),
            text=self.text,
            type=self.type,
            box_group=self.box_group.to_json() if self.box_group else None,
        )


@dataclass
class DocBoxGroup(Annotation):
    box_group: BoxGroup
    type: Optional[str] = None

    def to_json(self) -> Dict:
        return dict(
            _type="DocBoxGroup",  # Used for differenting between DocSpan and DocBox when loading the json
            box_group=self.box_group.to_json(),
            type=self.type
        )




# TODO[kylel] -- Implement
@dataclass
class Indexer:
    """Stores an index for a particular collection of Annotations.
    Indexes in this library focus on *INTERSECT* relations."""
    raise NotImplementedError


@dataclass
class DocSpanGroupIndexer(Indexer):

    num_pages: int

    # TODO[kylel] -- explain why?
    def __post_init__(self):
        self._index: List[IntervalTree] = [IntervalTree() for _ in range(self.num_pages)]

    def __getitem__(self, indices):
        page_id, annotation_slice = indices
        return [interval.data for interval in self._index[page_id][annotation_slice]]

    def index(self, doc_span_group: DocSpanGroup):
        all_matched_span_groups = []
        # TODO: does this work?  isnt it `for span in anno.span_group` and require `span_group` to implement `__iter__`
        for span in doc_span_group.span_group:
            for matched_span_group in self._index[span.page_id][span.start : span.end]:
                if matched_span_group not in all_matched_span_groups: # Deduplicate
                    all_matched_span_groups.append(matched_span_group)
        return all_matched_span_groups
