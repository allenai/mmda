"""

@rauthur

"""

import io
import os
import xml.etree.ElementTree as et
from typing import List, Optional, Text

import requests
from mmda.parsers.parser import Parser
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import SpanGroup
from mmda.types.document import Document
from mmda.types.names import Symbols
from mmda.types.span import Span

# processCitationList available in Grobid 0.7.1-SNAPSHOT and later
DEFAULT_API = "http://localhost:8070/api/processCitation"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _post_document(citations: str, url: str = DEFAULT_API) -> str:
    req = requests.post(url, data={"citations": citations, "includeRawCitations": "1"})

    if req.status_code != 200:
        raise RuntimeError(f"Unable to process citations. Received {req.status_code}!")

    return req.text


def get_title(citations: str, url: str = DEFAULT_API) -> Optional[str]:
    xml = _post_document(citations, url)
    root = et.parse(io.StringIO(xml)).getroot()
    matches = root.findall(".//title")

    if len(matches) == 0:
        return None

    if not matches[0].text:
        return None

    return matches[0].text.strip()
