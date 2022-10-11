"""



"""


from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
import warnings
import numpy as np


def is_overlap_1d(start1: float, end1: float, start2: float, end2: float, x: float = 0) -> bool:
    """Return whether two 1D intervals overlaps given x"""
    assert start1 <= end1
    assert start2 <= end2

    return not (start1 - x > end2 or start1 > end2 + x or end1 + x < start2 or end1 < start2 - x)  # ll  # rr


@dataclass
class Box:
    l: float
    t: float
    w: float
    h: float
    page: int

    def to_json(self) -> Dict[str, float]:
        return {'left': self.l, 'top': self.t, 'width': self.w, 'height': self.h, 'page': self.page}

    @classmethod
    def from_json(cls, box_dict: Dict[str, Union[float, int]]) -> "Box":
        return Box(l=box_dict['left'], t=box_dict['top'], w=box_dict['width'], h=box_dict['height'],
                   page=box_dict['page'])

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

        _x1, _x2 = np.clip([x1, x2], 0, page_width)
        _y1, _y2 = np.clip([y1, y2], 0, page_height)

        if _x2 < _x1:
            _x2 = _x1
        if _y2 < _y1:
            _y2 = _y1
        if (_x1, _y1, _x2, _y2) != (x1, y1, x2, y2):
            warnings.warn(
                f"The coordinates ({x1}, {y1}, {x2}, {y2}) are not valid and converted to ({_x1}, {_y1}, {_x2}, {_y2})."
            )

        return cls(_x1, _y1, _x2 - _x1, _y2 - _y1, page)

    @classmethod
    def small_boxes_to_big_box(cls, boxes: List["Box"]) -> "Box":
        """Computes one big box that tightly encapsulates all smaller input boxes"""
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            return None

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

    def is_overlap(self, other: "Box", x: float = 0.0, y: float = 0) -> bool:
        """
        Whether self overlaps with the other Box object.
        x, y distances
        """
        x11, y11, x12, y12 = self.coordinates
        x21, y21, x22, y22 = other.coordinates

        return is_overlap_1d(x11, x12, x21, x22, x) and is_overlap_1d(y11, y12, y21, y22, y)
