"""

@rauthur

"""

import io
import xml.etree.ElementTree as et
from typing import Optional

import requests

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
