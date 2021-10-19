from dataclasses import dataclass
from typing import Optional

import requests

_API_FIELDS = ["title", "abstract", "url"]
_API_URL = "https://api.semanticscholar.org/graph/v1/paper/{}?fields={}"


@dataclass
class PaperMetadata:
    id: str
    url: str
    title: str
    abstract: Optional[str]


def get_paper_metadata(paper_id: str, fields=_API_FIELDS) -> PaperMetadata:
    qs = ",".join(fields)
    url = _API_URL.format(paper_id, qs)

    req = requests.get(url)
    if req.status_code != 200:
        raise RuntimeError(f"Unable to retrieve paper: {paper_id}!")

    data = req.json()
    return PaperMetadata(
        id=data["paperId"],
        url=data["url"],
        title=data["title"],
        abstract=data["abstract"],
    )
