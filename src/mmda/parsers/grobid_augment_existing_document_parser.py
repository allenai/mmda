"""

@geli-gel

"""
from grobid_client.grobid_client import GrobidClient
from typing import Optional, List

import os
import xml.etree.ElementTree as et

from mmda.parsers.parser import Parser
from mmda.types import Metadata
from mmda.types.annotation import BoxGroup, Box
from mmda.types.document import Document
from mmda.types.names import PagesField, RowsField, TokensField

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

        bib_entries = self._get_grobid_bib_box_groups(xml_root)
        
        # note for if/when adding in relations between mention sources and bib targets:
        # big_entries metadata contains original grobid id attached to the BoxGroup.
        doc.annotate(
            bib_entries=bib_entries
            )

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

    def _get_grobid_bib_box_groups(self, root: et.Element) -> List[BoxGroup]:
        bib_list_root = root.find(".//tei:listBibl", NS)

        page_size_root = root.find(".//tei:facsimile", NS)
        page_size_data = page_size_root.findall(".//tei:surface", NS)
        page_sizes = dict()
        for data in page_size_data:
            page_sizes[int(data.attrib["n"]) - 1] = [float(data.attrib["lrx"]), float(data.attrib["lry"])]
        
        grobid_bibs = []
        bib_structs = bib_list_root.findall(".//tei:biblStruct", NS)
        for bib in bib_structs:
            coords_string = bib.attrib["coords"]
            boxes = self._xml_coords_to_boxes(coords_string, page_sizes)
            grobid_id = bib.attrib["{http://www.w3.org/XML/1998/namespace}id"]

            grobid_bibs.append(
                BoxGroup(
                boxes=boxes, 
                metadata=Metadata(grobid_id=grobid_id)
                )
            )

        return grobid_bibs
    
