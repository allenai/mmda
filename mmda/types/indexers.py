"""

Indexes for Annotations

"""

from typing import List

from abc import abstractmethod
from dataclasses import dataclass, field

from intervaltree import IntervalTree
from mmda.types.annotation import SpanGroup, Annotation


@dataclass
class Indexer:
    """Stores an index for a particular collection of Annotations.
    Indexes in this library focus on *INTERSECT* relations."""

    @abstractmethod
    def find(self, query: Annotation) -> List[Annotation]:
        """Returns all matching Annotations given a suitable query"""


@dataclass
class SpanGroupIndexer(Indexer):

    # careful; if write it as _index = IntervalTree(), all SpanGroupIndexers will share the same _index object
    _index: IntervalTree = field(default_factory=IntervalTree)

    # TODO[kylel] - maybe have more nullable args for different types of queryes (just start/end ints, just SpanGroup)
    def find(self, query: SpanGroup) -> List[SpanGroup]:
        if not isinstance(query, SpanGroup):
            raise ValueError(f'SpanGroupIndexer only works with `query` that is SpanGroup type')

        all_matched_span_groups = []
        for span in query.spans:
            for matched_span_group in self._index[span.start : span.end]:
                if matched_span_group.data not in all_matched_span_groups: # Deduplicate
                    all_matched_span_groups.append(matched_span_group.data)
        # retrieval can be out of order, so sort
        return sorted(all_matched_span_groups)

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key, value):
        self._index[key] = value