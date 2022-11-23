import csv
import io
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pytesseract
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import BoxGroup
from mmda.types.box import Box
from mmda.types.document import Document
from mmda.types.names import ImagesField


@dataclass
class TesseractData:
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    left: float
    top: float
    width: float
    height: float
    conf: int
    text: str

    def lrtb(self) -> Tuple[float, float, float, float]:
        return self.left, self.left + self.width, self.top, self.top + self.height

    @classmethod
    def from_csv(cls, line: Dict[str, str]) -> "TesseractData":
        return TesseractData(
            block_num=int(line["block_num"]),
            par_num=int(line["par_num"]),
            line_num=int(line["line_num"]),
            word_num=int(line["word_num"]),
            left=float(line["left"]),
            top=float(line["top"]),
            width=float(line["width"]),
            height=float(line["height"]),
            conf=int(line["conf"]),
            text=line["text"],
        )


class TesseractBlockPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["pytesseract"]
    REQUIRED_DOCUMENT_FIELDS = [ImagesField]

    def predict(self, document: Document) -> Iterable[BoxGroup]:
        box_groups = []

        for idx, image in enumerate(document.images):
            data = pytesseract.image_to_data(image)
            reader = csv.DictReader(io.StringIO(data), delimiter="\t")

            # Gather up lines that have any text in them
            parsed = [
                TesseractData.from_csv(line)
                for line in reader
                if len(line["text"].strip()) > 0
            ]
            # Discard boxes that are the entire page
            parsed = [
                p for p in parsed if p.width < image.width or p.height < image.height
            ]

            # TODO: Also include left, top in sort?
            parsed = sorted(parsed, key=lambda x: x.block_num)

            for key, blocks in itertools.groupby(parsed, key=lambda x: x.block_num):
                min_l = float("inf")
                max_r = float("-inf")
                min_t = float("inf")
                max_b = float("-inf")

                for block in blocks:
                    l, r, t, b = block.lrtb()

                    if l < min_l:
                        min_l = l
                    if r > max_r:
                        max_r = r
                    if t < min_t:
                        min_t = t
                    if b > max_b:
                        max_b = b

                w = max_r - min_l
                h = max_b - min_t

                box_groups.append(
                    BoxGroup(
                        id=key,
                        boxes=[
                            Box(l=min_l, t=min_t, w=w, h=h, page=idx).get_relative(
                                image.width, image.height
                            )
                        ],
                    )
                )

        return box_groups
