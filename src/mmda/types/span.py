"""



"""

from typing import List, Optional, Dict
from dataclasses import dataclass

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
        box_dict = span_dict.get('box')
        if box_dict:
            box = Box.from_json(box_dict=span_dict['box'])
        else:
            box = None
        return Span(start=span_dict['start'], end=span_dict['end'], box=box)

    def __lt__(self, other: 'Span'):
        return self.start < other.start

    def __contains__(self, item: 'Span'):
        return self.start <= item.start and self.end >= item.end

    @classmethod
    def small_spans_to_big_span(cls, spans: List['Span']) -> 'Span':
        # TODO: add warning for unsorted spans or not-contiguous spans
        # TODO: what happens when Boxes cant be merged?
        start = None
        end = None
        for span in spans:
            if span.start < start:
                start = span.start
            if span.end > end:
                end = span.end
        return Span(
            start=start,
            end=end,
            box=Box.small_boxes_to_big_box(boxes=[span.box for span in spans])
        )
