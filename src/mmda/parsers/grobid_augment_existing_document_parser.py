"""

@geli-gel

"""
from collections import defaultdict
from grobid_client.grobid_client import GrobidClient
from typing import Dict, Iterable, List, Optional, Tuple

import os
import xml.etree.ElementTree as et

from mmda.parsers.parser import Parser
from mmda.types import Metadata
from mmda.types.annotation import BoxGroup, Box, SpanGroup
from mmda.types.document import Document
from mmda.types.names import PagesField, RowsField, TokensField
from mmda.utils.tools import box_groups_to_span_groups

REQUIRED_DOCUMENT_FIELDS = [PagesField, RowsField, TokensField]
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidAugmentExistingDocumentParser(Parser):
    """Grobid parser that uses Grobid python client to hit a running
     Grobid server and convert resulting grobid XML TEI coordinates into 
     MMDA BoxGroups to annotate an existing Document.
     
     Run a Grobid server (from https://grobid.readthedocs.io/en/latest/Grobid-docker/):
     > docker pull lfoppiano/grobid:0.7.2
     > docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
    """

    def __init__(self, config_path: str = "grobid.config", check_server: bool = True):

        self.client = GrobidClient(config_path=config_path, check_server=check_server)

    def parse(self, input_pdf_path: str, doc: Document, xml_out_dir: Optional[str] = None) -> Document:

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
            xmlfile = os.path.join(xml_out_dir, os.path.basename(input_pdf_path).replace('.pdf', '.xml'))
            with open(xmlfile, 'w') as f_out:
                f_out.write(xml)

        self._parse_xml_onto_doc(xml, doc)
        return doc

    def _parse_xml_onto_doc(self, xml: str, doc: Document) -> Document:
        xml_root = et.fromstring(xml)
        self._cache_page_sizes(xml_root)

        # authors
        author_box_groups = self._get_box_groups(xml_root, "sourceDesc", "persName")
        doc.annotate(authors=box_groups_to_span_groups(author_box_groups, doc, center=True))

        # bibliography entries
        bib_entry_box_groups = self._get_box_groups(xml_root, "listBibl", "biblStruct")
        doc.annotate(bib_entries=box_groups_to_span_groups(bib_entry_box_groups, doc, center=True))

        # citation mentions
        citation_mention_box_groups = self._get_box_groups(xml_root, "body", "ref", type_attr="bibr")
        doc.annotate(citation_mentions=box_groups_to_span_groups(citation_mention_box_groups, doc, center=True))

        return doc

    def _xml_coords_to_boxes(self, coords_attribute: str):
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
                ).get_relative(*self.page_sizes[proper_page])
            )
        return boxes

    def _cache_page_sizes(self, root: et.Element):
        page_size_root = root.find(".//tei:facsimile", NS)
        page_size_data = page_size_root.findall(".//tei:surface", NS)
        page_sizes = dict()
        for data in page_size_data:
            page_sizes[int(data.attrib["n"]) - 1] = [float(data.attrib["lrx"]), float(data.attrib["lry"])]
        self.page_sizes = page_sizes
        return page_sizes

    def _get_box_groups(
            self,
            root: et.Element,
            list_tag: str,
            item_tag: str,
            type_attr: Optional[str] = None
    ) -> List[BoxGroup]:
        item_list_root = root.find(f".//tei:{list_tag}", NS)

        box_groups = []
        if type_attr:
            elements = item_list_root.findall(f".//tei:{item_tag}[@type='{type_attr}']", NS)
        else:
            elements = item_list_root.findall(f".//tei:{item_tag}", NS)

        for e in elements:
            coords_string = e.attrib["coords"]
            boxes = self._xml_coords_to_boxes(coords_string)

            box_groups.append(BoxGroup(boxes=boxes))

        return box_groups
