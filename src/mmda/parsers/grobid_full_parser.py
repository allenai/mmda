"""

@geli-gel

"""
from collections import defaultdict
import json
import re
from tempfile import NamedTemporaryFile
from grobid_client.grobid_client import GrobidClient
from typing import Dict, Optional, List

import os
import xml.etree.ElementTree as et

from mmda.parsers.parser import Parser
from mmda.types import Metadata
from mmda.types.annotation import BoxGroup, Box
from mmda.types.document import Document
from mmda.types.names import PagesField, RowsField, TokensField
from mmda.utils.tools import box_groups_to_span_groups

REQUIRED_DOCUMENT_FIELDS = [PagesField, RowsField, TokensField]
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def make_variable_name(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_]', "_", s)
    if s[0].isdigit():
        s = "_" + s
    return s


class GrobidFullParser(Parser):
    """Grobid parser that uses Grobid python client to hit a running
     Grobid server and convert resulting grobid XML TEI coordinates into
     MMDA BoxGroups to annotate an existing Document.

     Run a Grobid server (from https://grobid.readthedocs.io/en/latest/Grobid-docker/):
     > docker pull lfoppiano/grobid:0.7.2
     > docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
    """

    def __init__(
        self,
        grobid_config: Optional[dict] = None,
        check_server: bool = True
    ):

        self.grobid_config = grobid_config or {
            "grobid_server": "http://localhost:8070",
            "batch_size": 1000,
            "sleep_time": 5,
            "timeout": 60,
            "coordinates": [
                "figure", "ref", "biblStruct", "formula", "s", "head", "p"
            ]
        }
        assert "coordinates" in self.grobid_config, \
            "Grobid config must contain 'coordinates' key"

        with NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(self.grobid_config, f)
            config_path = f.name

        self.client = GrobidClient(
            config_path=config_path,
            check_server=check_server
        )

        os.remove(config_path)

    def parse(
        self,
        input_pdf_path: str,
        doc: Document,
        xml_out_dir: Optional[str] = None
    ) -> Document:

        assert doc.symbols != ""
        for field in REQUIRED_DOCUMENT_FIELDS:
            assert field in doc.fields

        (_, _, xml) = self.client.process_pdf(
                "processFulltextDocument",
                input_pdf_path,
                generateIDs=False,
                consolidate_header=False,
                consolidate_citations=False,
                include_raw_citations=False,
                include_raw_affiliations=False,
                tei_coordinates=True,
                segment_sentences=True
            )

        if xml_out_dir:
            os.makedirs(xml_out_dir, exist_ok=True)
            xmlfile = os.path.join(
                xml_out_dir,
                os.path.basename(input_pdf_path).replace('.pdf', '.xml')
            )
            with open(xmlfile, 'w') as f_out:
                f_out.write(xml)

        self._parse_xml_onto_doc(xml, doc)
        return doc

    def _parse_xml_onto_doc(self, xml: str, doc: Document) -> Document:
        xml_root = et.fromstring(xml)

        all_box_groups = self._get_box_groups(xml_root)
        for field, box_groups in all_box_groups.items():

            span_groups = []
            for box_group in box_groups:
                span_group, *others = box_groups_to_span_groups(
                    box_groups=[box_group], doc=doc, center=True
                )
                assert len(others) == 0, \
                    "BoxGroup should only contain one SpanGroup"
                span_group.metadata = box_group.metadata
                span_groups.append(span_group)

            # note for if/when adding in relations between mention sources and
            # bib targets: big_entries metadata contains original grobid id
            # attached to the BoxGroup.
            doc.annotate(**{field: span_groups})

        return doc

    def _xml_coords_to_boxes(self, coords_attribute: str, page_sizes: dict):
        coords_list = coords_attribute.split(";")
        boxes = []
        for coords in coords_list:
            pg, x, y, w, h = coords.split(",")
            proper_page = int(pg) - 1
            boxes.append(
                Box(
                    l=float(x),
                    t=float(y),
                    w=float(w),
                    h=float(h),
                    page=proper_page
                ).get_relative(*page_sizes[proper_page])
            )
        return boxes

    def _get_box_groups(self, root: et.Element) -> Dict[str, List[BoxGroup]]:
        page_size_root = root.find(".//tei:facsimile", NS)
        assert page_size_root is not None, "No facsimile found in Grobid XML"

        page_size_data = page_size_root.findall(".//tei:surface", NS)
        page_sizes = dict()
        for data in page_size_data:
            page_sizes[int(data.attrib["n"]) - 1] = [
                float(data.attrib["lrx"]), float(data.attrib["lry"])
            ]

        all_boxes: Dict[str, List[BoxGroup]] = defaultdict(list)

        for field in self.grobid_config["coordinates"]:
            structs = root.findall(f".//tei:{field}", NS)
            for i, struct in enumerate(structs):
                coords_string = struct.attrib.pop("coords", None)
                if coords_string is None:
                    continue
                boxes = self._xml_coords_to_boxes(coords_string, page_sizes)
                metadata = Metadata(
                    grobid_order=i,
                    **{
                        make_variable_name(k): v
                        for k, v in struct.attrib.items()
                    }
                )
                box_group = BoxGroup(boxes=boxes, metadata=metadata)
                all_boxes[field].append(box_group)

        return all_boxes


if __name__ == "__main__":
    from mmda.parsers import PDFPlumberParser
    doc = PDFPlumberParser().parse("/Users/lucas/local_scim/tests/test_1.pdf")
    doc = GrobidFullParser().parse("/Users/lucas/local_scim/tests/test_1.pdf", doc)
    print(doc.fields)
    import ipdb; ipdb.set_trace()
