import json
import random
import string
from dataclasses import dataclass
from typing import Protocol

from mmda.eval import s2
from mmda.eval.metrics import levenshtein
from mmda.parsers.grobid_parser import GrobidHeaderParser


@dataclass(frozen=True)
class LabeledDoc:
    id: str
    title: str
    abstract: str
    url: str


class PredictedDoc(Protocol):
    @property
    def title(self):
        raise NotImplementedError

    @property
    def abstract(self):
        raise NotImplementedError


@dataclass
class DefaultPredictedDoc:
    id: str
    title: str
    abstract: str


def grobid_prediction(pdf_path: str, parser: GrobidHeaderParser) -> PredictedDoc:
    doc = parser.parse(pdf_path)

    title = " ".join(doc.title[0].symbols)
    abstract = "\n".join([" ".join(x.symbols) for x in doc.abstract])

    return DefaultPredictedDoc(id=pdf_path, title=title, abstract=abstract)


def s2_prediction(id_: str) -> PredictedDoc:
    metadata = s2.get_paper_metadata(id_)

    title = metadata.title if metadata.title else ""
    abstract = metadata.abstract if metadata.abstract else ""

    return DefaultPredictedDoc(id=id_, title=title, abstract=abstract)


def random_prediction(label: LabeledDoc) -> PredictedDoc:
    """
    Jumbled ASCII lowercase characters of same length as title/abstract
    """

    def rand_str(n: int) -> str:
        return "".join([random.choice(string.ascii_lowercase) for _ in range(n)])

    random_title = rand_str(len(label.title))
    random_abstract = rand_str(len(label.abstract))

    return DefaultPredictedDoc(
        id=label.id, title=random_title, abstract=random_abstract
    )


def read_labels(labels_json_path: str) -> list[LabeledDoc]:
    """Read label JSON data into expected format for VLUE evaluation.

    Args:
        labels_json_path (str): Path to curated labels JSON file

    Returns:
        list[LabeledDoc]: List of labeled documents
    """
    with open(labels_json_path, encoding="utf-8") as f:
        labels = [LabeledDoc(**l) for l in json.loads(f.read())]

    return labels


def score(label: LabeledDoc, pred: PredictedDoc, attr: str) -> float:
    """Evaluate a prediction for a specific field on VLUE.

    Args:
        label (LabeledDoc): Label read from JSON file
        pred (PredictedDoc): Predicted title/abstract document
        attr (str): Which field to evaluate

    Returns:
        float: Score between 0 and 1.
    """
    a = label.__getattribute__(attr)
    b = pred.__getattribute__(attr)

    return 1 - levenshtein(a, b, case_sensitive=True) / max(len(a), len(b))
