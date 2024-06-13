"""



"""


import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np


def _is_overlap_1d(
    start1: float, end1: float, start2: float, end2: float, x: float = 0
) -> bool:
    """Return whether two 1D intervals overlaps given x"""
    assert start1 <= end1
    assert start2 <= end2

    return not (
        start1 - x > end2 or start1 > end2 + x or end1 + x < start2 or end1 < start2 - x
    )  # ll  # rr


class Box:
    def __init__(self, l: float, t: float, w: float, h: float, page: int) -> None:
        if w < 0 or h < 0:
            raise ValueError(f"Width and height must be non-negative, got {w} and {h}")
        if page < 0:
            raise ValueError(f"Page must be non-negative, got {page}")
        if l < 0 or t < 0:
            raise ValueError(f"Left and top must be non-negative, got {l} and {t}")
        self.l = l
        self.t = t
        self.w = w
        self.h = h
        self.page = page

    def to_json(self) -> Dict[str, float]:
        return {
            "left": self.l,
            "top": self.t,
            "width": self.w,
            "height": self.h,
            "page": self.page,
        }

    @classmethod
    def from_json(cls, box_dict: Dict[str, Union[float, int]]) -> "Box":
        return Box(
            l=box_dict["left"],
            t=box_dict["top"],
            w=box_dict["width"],
            h=box_dict["height"],
            page=box_dict["page"],
        )

    @classmethod
    def from_coordinates(cls, x1: float, y1: float, x2: float, y2: float, page: int):
        return Box(l=x1, t=y1, w=x2 - x1, h=y2 - y1, page=page)

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

    def is_overlap(
        self, other: "Box", x: float = 0.0, y: float = 0, center: bool = False
    ) -> bool:
        """
        Whether self overlaps with the other Box object.
        x, y distances for padding
        center (bool) if True, only consider overlapping if this box's center is contained by other
        """
        if self.page != other.page:
            return False

        x11, y11, x12, y12 = self.coordinates
        x21, y21, x22, y22 = other.coordinates
        if center:
            center_x, center_y = self.center
            res = is_overlap_1d(center_x, center_x, x21, x22, x) and is_overlap_1d(
                center_y, center_y, y21, y22, y
            )
        else:
            res = is_overlap_1d(x11, x12, x21, x22, x) and is_overlap_1d(
                y11, y12, y21, y22, y
            )
        return res

    @classmethod
    def cluster_boxes(cls, boxes: List["Box"]) -> List[List[int]]:
        """
        Cluster boxes into groups based on any overlap.
        """
        if not boxes:
            return []

        clusters: List[List[int]] = [[0]]
        cluster_id_to_big_box: Dict[int, Box] = {0: boxes[0]}
        for box_id in range(1, len(boxes)):
            box = boxes[box_id]

            # check all the clusters to see if the box overlaps with any of them
            is_overlap = False
            for cluster_id, big_box in cluster_id_to_big_box.items():
                if box.is_overlap(big_box, x=0, y=0):
                    is_overlap = True
                    break

            # resolve
            if is_overlap:
                clusters[cluster_id].append(box_id)
                cluster_id_to_big_box[cluster_id] = cls.small_boxes_to_big_box(
                    [box, big_box]
                )
            else:
                clusters.append([box_id])
                cluster_id_to_big_box[len(clusters) - 1] = box

        # sort clusters
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key=lambda x: x[0])

        return clusters

    def shrink(self, delta: float, ignore: bool = True, clip: bool = True):
        x1, y1, x2, y2 = self.coordinates
        if x2 - x1 <= 2 * delta:
            if ignore:
                logging.warning(f"box's x-coords {self} shrink too much. Ignoring.")
            else:
                raise ValueError(
                    f"box's x-coords {self} shrink too much with delta={delta}."
                )
        else:
            if clip:
                logging.warning(
                    f"box's x-coords {self} go beyond page boundary. Clipping..."
                )
                x1 = min(x1 + delta, 1.0)
                x2 = max(x2 - delta, 0.0)
            else:
                raise ValueError(
                    f"box's x-coordinates {self} go beyond page boundary. need clip."
                )

        if y2 - y1 <= 2 * delta:
            if ignore:
                logging.warning(f"box's y-coords {self} shrink too much. Ignoring.")
            else:
                raise ValueError(
                    f"box's y-coords {self} shrink too much with delta={delta}."
                )
        else:
            if clip:
                logging.warning(
                    f"box's y-coords {self} go beyond page boundary. Clipping..."
                )
                y1 = min(y1 + delta, 1.0)
                y2 = max(y2 - delta, 0.0)
            else:
                raise ValueError(
                    f"box's y-coordinates {self} go beyond page boundary. need clip."
                )

        self.l = x1
        self.t = y1
        self.w = x2 - x1
        self.h = y2 - y1
