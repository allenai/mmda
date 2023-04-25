"""

Indexes for Annotations

"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from ncls import NCLS

from mmda.types.annotation import Annotation, Box, BoxGroup, SpanGroup


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
            pd.Series(ids, dtype=np.int64),
        )

        self._ensure_disjoint()

    def _ensure_disjoint(self) -> None:
        """
        Constituent span groups must be fully disjoint.
        Ensure the integrity of the built index.
        """
        for span_group in self._sgs:
            for span in span_group.spans:
                match_ids = [
                    matched_id
                    for _start, _end, matched_id in self._index.find_overlap(
                        span.start, span.end
                    )
                ]
                if len(match_ids) > 1:
                    matches = [self._sgs[match_id].to_json() for match_id in match_ids]
                    raise ValueError(
                        f"Detected overlap! While processing the Span {span} as part of query SpanGroup {span_group.to_json()}, we found that it overlaps with existing SpanGroup(s):\n"
                        + "\n".join(
                            [f"\t{i}\t{m} " for i, m in zip(match_ids, matches)]
                        )
                    )

    def find(self, query: SpanGroup) -> List[SpanGroup]:
        if not isinstance(query, SpanGroup):
            raise ValueError(
                f"SpanGroupIndexer only works with `query` that is SpanGroup type"
            )

        if not query.spans:
            return []

        matched_ids = set()

        for span in query.spans:
            for _start, _end, matched_id in self._index.find_overlap(
                span.start, span.end
            ):
                matched_ids.add(matched_id)

        matched_span_groups = [self._sgs[matched_id] for matched_id in matched_ids]

        # Retrieval above doesn't preserve document order; sort here
        # TODO: provide option to return matched span groups in same order as self._sgs
        #   (the span groups the index was built with originally)
        return sorted(list(matched_span_groups))


class BoxGroupIndexer(Indexer):
    """
    Manages a data structure for locating overlapping BoxGroups.
    Builds a static nested containment list from BoxGroups
    and accepts other BoxGroups as search probes.
    """

    def __init__(self, box_groups: List[BoxGroup]) -> None:
        self._box_id_to_box_group_id = {}

        self._bgs = box_groups
        self._boxes = []
        box_id = 0
        for bg_id, bg in enumerate(box_groups):
            for box in bg.boxes:
                self._boxes.append(box)
                self._box_id_to_box_group_id[box_id] = bg_id
                box_id += 1

        self._np_boxes_x1 = np.array([b.l for b in self._boxes])
        self._np_boxes_y1 = np.array([b.t for b in self._boxes])
        self._np_boxes_x2 = np.array([b.l + b.w for b in self._boxes])
        self._np_boxes_y2 = np.array([b.t + b.h for b in self._boxes])

        self._ensure_disjoint()

    def _find_overlap_boxes(self, query: Box) -> List[int]:
        x1, y1, x2, y2 = query.coordinates
        mask = (
            (self._np_boxes_x1 <= x2)
            & (self._np_boxes_x2 >= x1)
            & (self._np_boxes_y1 <= y2)
            & (self._np_boxes_y2 >= y1)
        )
        return np.where(mask)[0].tolist()

    def _find_overlap_box_groups(self, query: Box) -> List[int]:
        return [
            self._box_id_to_box_group_id[box_id]
            for box_id in self._find_overlap_boxes(query)
        ]

    def _ensure_disjoint(self) -> None:
        """
        Constituent box groups must be fully disjoint.
        Ensure the integrity of the built index.
        """
        for box_group in self._bgs:
            for box in box_group.boxes:
                match_ids = self._find_overlap_box_groups(query=box)
                if len(match_ids) > 1:
                    matches = [self._bgs[match_id].to_json() for match_id in match_ids]
                    raise ValueError(
                        f"Detected overlap! While processing the Box {box} as part of query BoxGroup {box_group.to_json()}, we found that it overlaps with existing BoxGroup(s):\n"
                        + "\n".join(
                            [f"\t{i}\t{m} " for i, m in zip(match_ids, matches)]
                        )
                    )

    def find(self, query: BoxGroup) -> List[BoxGroup]:
        if not isinstance(query, BoxGroup):
            raise ValueError(
                f"BoxGroupIndexer only works with `query` that is BoxGroup type"
            )

        if not query.boxes:
            return []

        match_ids = []
        for box in query.boxes:
            match_ids.extend(self._find_overlap_box_groups(query=box))

        return [self._bgs[match_id] for match_id in sorted(set(match_ids))]
