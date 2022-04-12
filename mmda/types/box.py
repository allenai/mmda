"""



"""


from typing import List, Optional, Dict, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field
import warnings


def is_overlap_1d(start1: float, end1: float, start2: float, end2: float) -> bool:
    """Return whether two 1D intervals overlaps"""
    assert start1 <= end1
    assert start2 <= end2

    return not (start1 > end2 or end1 < start2)  # ll  # rr


@dataclass
class Box:
    l: float
    t: float
    w: float
    h: float
    page: int

    def to_json(self) -> List:
        return [self.l, self.t, self.w, self.h, self.page]
        # return dict(l=self.l, t=self.t, w=self.w, h=self.h, page=self.page)

    @classmethod
    def from_json(cls, box_coords: List) -> "Box":
        l, t, w, h, page = box_coords
        return Box(l=l, t=t, w=w, h=h, page=page)
        # return Box(l=box_coords['l'], t=box_coords['t'], w=box_coords['w'], h=box_coords['h'], page=box_coords['page'])

    @classmethod
    def from_coordinates(cls, x1: float, y1: float, x2: float, y2: float, page: int):
        return cls(x1, y1, x2 - x1, y2 - y1, page)

    @classmethod
    def from_pdf_coordinates(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        page_width: float,
        page_height: float,
        page: int,
    ):
        """
        Convert PDF coordinates to absolute coordinates.
        The difference between from_pdf_coordinates and from_coordinates is that this function
        will perform extra checks to ensure the coordinates are valid, i.e.,
        0<= x1 <= x2 <= page_width and 0<= y1 <= y2 <= page_height.
        """
        error_msg = []
        if x1 > x2:
            error_msg.append(f"Box Coordinates x1 > x2: {x1} > {x2}, enforcing x2=x1")
            x2 = x1
        if y1 > y2:
            error_msg.append(f"Box Coordinates y1 > y2: {y1} > {y2}, enforcing y2=y1")
            y2 = y1

        if x1 < 0:
            error_msg.append(f"Box Coordinates x1 < 0: {x1}, enforcing x1=0")
            x1 = 0
        if y1 < 0:
            error_msg.append(f"Box Coordinates y1 < 0: {y1}, enforcing y1=0")
            y1 = 0
        if x2 > page_width:
            error_msg.append(
                f"Box Coordinates x2 > page_width: {x2} > {page_width}, enforcing x2=page_width"
            )
            x2 = page_width
        if y2 > page_height:
            error_msg.append(
                f"Box Coordinates y2 > page_height: {y2} > {page_height}, enforcing y2=page_height"
            )
            y2 = page_height

        if len(error_msg) > 0:
            warnings.warn("\n".join(["Issues for initializing Box:"]+error_msg))

        return cls(x1, y1, x2 - x1, y2 - y1, page)

    @classmethod
    def small_boxes_to_big_box(cls, boxes: List["Box"]) -> "Box":
        """Computes one big box that tightly encapsulates all smaller input boxes"""
        if len({box.page for box in boxes}) != 1:
            raise ValueError(f"Bboxes not all on same page: {boxes}")
        x1 = min([bbox.l for bbox in boxes])
        y1 = min([bbox.t for bbox in boxes])
        x2 = max([bbox.l + bbox.w for bbox in boxes])
        y2 = max([bbox.t + bbox.h for bbox in boxes])
        return Box(page=boxes[0].page, l=x1, t=y1, w=x2 - x1, h=y2 - y1)

    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        """Return a tuple of the (x1, y1, x2, y2) format."""
        return self.l, self.t, self.l + self.w, self.t + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.l + self.w / 2, self.t + self.h / 2

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        """Return a tuple of the (left, top, width, height) format."""
        return self.l, self.t, self.w, self.h

    def get_relative(self, page_width: float, page_height: float) -> "Box":
        """Get the relative coordinates of self based on page_width, page_height."""
        return self.__class__(
            l=float(self.l) / page_width,
            t=float(self.t) / page_height,
            w=float(self.w) / page_width,
            h=float(self.h) / page_height,
            page=self.page,
        )

    def get_absolute(self, page_width: int, page_height: int) -> "Box":
        """Get the absolute coordinates of self based on page_width, page_height."""
        return self.__class__(
            l=self.l * page_width,
            t=self.t * page_height,
            w=self.w * page_width,
            h=self.h * page_height,
            page=self.page,
        )

    def is_overlap(self, other: "Box") -> bool:
        """Whether self overlaps with the other Box object."""
        x11, y11, x12, y12 = self.coordinates
        x21, y21, x22, y22 = other.coordinates

        return is_overlap_1d(x11, x12, x21, x22) and is_overlap_1d(y11, y12, y21, y22)
