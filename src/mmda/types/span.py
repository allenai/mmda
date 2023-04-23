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
    def small_spans_to_big_span(cls, spans: List["Span"]) -> "Span":
        # TODO: add warning for unsorted spans or not-contiguous spans
        # TODO: what happens when Boxes cant be merged?
        start = min([span.start for span in spans])
        end = max([span.end for span in spans])
        return Span(start=start, end=end)

    def is_overlap(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

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
                    [span, big_span]
                )
            else:
                clusters.append([span_id])
                cluster_id_to_big_span[len(clusters) - 1] = span

        # sort clusters
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key=lambda x: x[0])

        return clusters
    