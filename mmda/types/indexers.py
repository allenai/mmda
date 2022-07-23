"""

Indexes for Annotations

"""

from typing import List

from abc import abstractmethod
from dataclasses import dataclass, field

from mmda.types.annotation import SpanGroup, Annotation
from ncls import NCLS
import numpy as np
import pandas as pd


@dataclass
class Indexer:
    """Stores an index for a particular collection of Annotations.
    Indexes in this library focus on *INTERSECT* relations."""

    @abstractmethod
    def find(self, query: Annotation) -> List[Annotation]:
        """Returns all matching Annotations given a suitable query"""
        raise NotImplementedError()


class SpanGroupIndexer(Indexer):
    """
    Manages a data structure for locating overlapping SpanGroups.
    Builds a static nested containment list from SpanGroups
    and accepts other SpanGroups as search probes.

    See: https://github.com/biocore-ntnu/ncls

    [citation]
    Alexander V. Alekseyenko, Christopher J. Lee;
    Nested Containment List (NCList): a new algorithm for accelerating interval query of genome
      alignment and interval databases, Bioinformatics,
    Volume 23, Issue 11, 1 June 2007, Pages 1386–1393, https://doi.org/10.1093/bioinformatics/btl647
    """

    def __init__(self, span_groups: List[SpanGroup]) -> None:
        starts = []
        ends = []
        ids = []

        for sg_id, span_group in enumerate(span_groups):
            for span in span_group.spans:
                starts.append(span.start)
                ends.append(span.end)
                ids.append(sg_id)

        self._sgs = span_groups
        self._index = NCLS(
            pd.Series(starts, dtype=np.int64),
            pd.Series(ends, dtype=np.int64),
            pd.Series(ids, dtype=np.int64)
        )

        self._ensure_disjoint()

    def _ensure_disjoint(self) -> None:
        """
        Constituent span groups must be fully disjoint.
        Ensure the integrity of the built index.
        """
        for span_group in self._sgs:
            for span in span_group.spans:
                matches = [match for match in self._index.find_overlap(span.start, span.end)]
                if len(matches) > 1:
                    raise ValueError(
                        f"Detected overlap with existing SpanGroup(s) {matches} for {span_group}"
                    )

    def find(self, query: SpanGroup) -> List[SpanGroup]:
        if not isinstance(query, SpanGroup):
            raise ValueError(f'SpanGroupIndexer only works with `query` that is SpanGroup type')

        if not query.spans:
            return []

        matched_ids = set()

        for span in query.spans:
            for _start, _end, matched_id in self._index.find_overlap(span.start, span.end):
                matched_ids.add(matched_id)

        matched_span_groups = [self._sgs[matched_id] for matched_id in matched_ids]

        # Retrieval above doesn't preserve document order; sort here
        # TODO: provide option to return matched span groups in same order as self._sgs
        #   (the span groups the index was built with originally)
        return sorted(list(matched_span_groups))


