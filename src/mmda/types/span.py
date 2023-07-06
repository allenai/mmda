"""



"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from mmda.types.box import Box


@dataclass
class Span:
    start: int
    end: int
    box: Optional[Box] = None

    def to_json(self) -> Dict:
        if self.box:
            return dict(start=self.start, end=self.end, box=self.box.to_json())
        else:
            return dict(start=self.start, end=self.end)

    @classmethod
    def from_json(cls, span_dict) -> "Span":
        box_dict = span_dict.get("box")
        if box_dict:
            box = Box.from_json(box_dict=span_dict["box"])
        else:
            box = None
        return Span(start=span_dict["start"], end=span_dict["end"], box=box)

    def __lt__(self, other: "Span"):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start

    @classmethod
    def small_spans_to_big_span(
        cls, spans: List["Span"], merge_boxes: bool = True
    ) -> "Span":
        # TODO: add warning for non-contiguous spans?
        start = spans[0].start
        end = spans[0].end
        for span in spans[1:]:
            if span.start < start:
                start = span.start
            if span.end > end:
                end = span.end
        if merge_boxes and all(span.box for span in spans):
            new_box = Box.small_boxes_to_big_box(boxes=[span.box for span in spans])
        else:
            new_box = None
        return Span(
            start=start,
            end=end,
            box=new_box,
        )

    @classmethod
    def cluster_spans(cls, spans: List["Span"]) -> List[List[int]]:
        """
        Cluster spans into groups based on any overlap.
        """
        if not spans:
            return []

        clusters: List[List[int]] = [[0]]
        cluster_id_to_big_span: Dict[int, Span] = {0: spans[0]}
        for span_id in range(1, len(spans)):
            span = spans[span_id]

            # check all the clusters to see if the span overlaps with any of them
            is_overlap = False
            for cluster_id, big_span in cluster_id_to_big_span.items():
                if span.is_overlap(big_span):
                    is_overlap = True
                    break

            # resolve
            if is_overlap:
                clusters[cluster_id].append(span_id)
                cluster_id_to_big_span[cluster_id] = cls.small_spans_to_big_span(
                    [span, big_span],
                    merge_boxes=False,
                )
            else:
                clusters.append([span_id])
                cluster_id_to_big_span[len(clusters) - 1] = span

        # sort clusters
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key=lambda x: x[0])

        return clusters

    def is_overlap(self, other: "Span") -> bool:
        is_self_before_other = self.start < other.end and self.end > other.start
        is_other_before_self = other.start < self.end and other.end > self.start
        return is_self_before_other or is_other_before_self

    @classmethod
    def are_disjoint(cls, spans: List["Span"]) -> bool:
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if spans[i].is_overlap(other=spans[j]):
                    return False
        return True
