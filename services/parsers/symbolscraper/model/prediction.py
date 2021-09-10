from typing import List

from pydantic import BaseModel, Field, conint, conlist, validator


_SPAN_BOX_LEN = 5


class Span(BaseModel):
    start: int
    end: int
    box: List[float]

    @validator("box")
    def check_number_of_elements_and_that_last_element_is_really_an_int_pretending_to_be_a_float(
        cls, box
    ):
        if len(box) != _SPAN_BOX_LEN:
            raise ValueError(f"box should have {_SPAN_BOX_LEN} things")
        if not box[_SPAN_BOX_LEN - 1].is_integer():
            raise ValueError("page should really be an int pretending to be a float")
        return box


class SpanGroup(BaseModel):
    spans: List[Span]
    id: int

    @validator("id")
    def id_is_a_mathematical_integer(cls, id_):
        if id_ < 0:
            raise ValueError("id of a span group should be an integer")
        return id_


class Prediction(BaseModel):
    symbols: str
    images: List[str]
    pages: List[SpanGroup]
    tokens: List[SpanGroup]
    rows: List[SpanGroup]
