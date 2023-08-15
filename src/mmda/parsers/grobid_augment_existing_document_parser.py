"""

@geli-gel

"""
from grobid_client.grobid_client import GrobidClient
from typing import List, Optional

import logging
import os
import xml.etree.ElementTree as et

from mmda.parsers.parser import Parser
from mmda.types import Metadata
from mmda.types.annotation import BoxGroup, Box, SpanGroup
from mmda.types.document import Document
from mmda.types.names import PagesField, RowsField, TokensField
from mmda.utils.tools import box_groups_to_span_groups

REQUIRED_DOCUMENT_FIELDS = [PagesField, RowsField, TokensField]
NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}
ID_ATTR_KEY = f"{{{NS['xml']}}}id"


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

        (_, status, xml) = self.client.process_pdf(
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
        if status == 200:
            if xml_out_dir:
                os.makedirs(xml_out_dir, exist_ok=True)
                xmlfile = os.path.join(
                    xml_out_dir, os.path.basename(input_pdf_path).replace(".pdf", ".xml")
                )
                with open(xmlfile, "w") as f_out:
                    f_out.write(xml)
            self._parse_xml_onto_doc(xml, doc)
        elif status == 408:
            raise TimeoutError("Grobid client timed out")
        else:
            raise RuntimeError(f"Grobid client failed with status {status}")
        return doc

    def _parse_xml_onto_doc(self, xml: str, doc: Document) -> Document:
        xml_root = et.fromstring(xml)
        self._cache_page_sizes(xml_root)

        # authors
        author_box_groups = self._get_box_groups(xml_root, "sourceDesc", "persName")
        doc.annotate(
            authors=box_groups_to_span_groups(author_box_groups, doc, center=True)
        )

        # bibliography entries
        bib_entry_box_groups = self._get_box_groups(xml_root, "listBibl", "biblStruct")
        doc.annotate(
            bib_entries=box_groups_to_span_groups(
                bib_entry_box_groups, doc, center=True
            )
        )

        # citation mentions
        citation_mention_box_groups = self._get_box_groups(
            xml_root, "body", "ref", type_attr="bibr"
        )
        doc.annotate(
            citation_mentions=box_groups_to_span_groups(
                citation_mention_box_groups, doc, center=True
            )
        )

        # sections
        # Grobid provides coordinates and number attributes for section headers, and coordinates for
        # sentences within the body text, also tagged by paragraphs.
        # We use these to annotate the document in order to provide a hierarchical structure:
        # e.g. doc.sections.header, doc.sections[0].paragraphs[0].sentences[0]
        section_box_groups, heading_box_groups, paragraph_box_groups, sentence_box_groups = \
            self._get_structured_body_text_box_groups(xml_root)
        doc.annotate(
            sections=box_groups_to_span_groups(
                section_box_groups, doc, center=True
            )
        )
        doc.annotate(
            headings=box_groups_to_span_groups(
                heading_box_groups, doc, center=False
            )
        )
        doc.annotate(
            paragraphs=box_groups_to_span_groups(
                paragraph_box_groups, doc, center=True
            )
        )
        doc.annotate(
            sentences=box_groups_to_span_groups(
                sentence_box_groups, doc, center=True
            )
        )

        return doc

    def _xml_coords_to_boxes(self, coords_attribute: str):
        coords_list = coords_attribute.split(";")
        boxes = []
        for coords in coords_list:
            try:
                pg, x, y, w, h = coords.split(",")
                proper_page = int(pg) - 1
                boxes.append(
                    Box(
                        l=float(x), t=float(y), w=float(w), h=float(h), page=proper_page
                    ).get_relative(*self.page_sizes[proper_page])
                )
            except ValueError:
                logging.warning(f"Could not parse coords: '{coords}'")
                pass
        return boxes

    def _cache_page_sizes(self, root: et.Element):
        page_size_root = root.find(".//tei:facsimile", NS)
        page_size_data = page_size_root.findall(".//tei:surface", NS)
        page_sizes = dict()
        for data in page_size_data:
            page_sizes[int(data.attrib["n"]) - 1] = [
                float(data.attrib["lrx"]),
                float(data.attrib["lry"]),
            ]
        self.page_sizes = page_sizes
        return page_sizes

    def _get_box_groups(
        self,
        root: et.Element,
        list_tag: str,
        item_tag: str,
        type_attr: Optional[str] = None,
    ) -> List[BoxGroup]:
        item_list_root = root.find(f".//tei:{list_tag}", NS)

        box_groups = []
        if type_attr:
            elements = item_list_root.findall(
                f".//tei:{item_tag}[@type='{type_attr}']", NS
            )
        else:
            elements = item_list_root.findall(f".//tei:{item_tag}", NS)

        for e in elements:
            coords_string = e.attrib["coords"]
            boxes = self._xml_coords_to_boxes(coords_string)

            grobid_id = e.attrib[ID_ATTR_KEY] if ID_ATTR_KEY in e.keys() else None
            target_id = e.attrib["target"][1:] if (item_tag == "ref" and "target" in e.keys()) else None

            if grobid_id and target_id:
                box_groups.append(
                    BoxGroup(
                        boxes=boxes,
                        metadata=Metadata(grobid_id=grobid_id, target_id=target_id),
                    )
                )
            elif grobid_id:
                box_groups.append(
                    BoxGroup(boxes=boxes, metadata=Metadata(grobid_id=grobid_id))
                )
            elif target_id:
                box_groups.append(
                    BoxGroup(boxes=boxes, metadata=Metadata(target_id=target_id))
                )
            else:
                box_groups.append(BoxGroup(boxes=boxes))
        return box_groups

    def _get_heading_box_group(
            self,
            section_div: et.Element
    ) -> Optional[BoxGroup]:
        box_group = None
        heading_element = section_div.find(f".//tei:head", NS)
        if heading_element is not None:  # elements evaluate as False if no children
            coords_string = heading_element.attrib["coords"]
            boxes = self._xml_coords_to_boxes(coords_string)
            number = heading_element.attrib["n"] if "n" in heading_element.keys() else None
            section_title = heading_element.text
            box_group = BoxGroup(
                boxes=boxes,
                metadata=Metadata(number=number, title=section_title),
            )
        return box_group

    def _get_structured_body_text_box_groups(
            self,
            root: et.Element
    ) -> (List[BoxGroup], List[BoxGroup], List[BoxGroup], List[BoxGroup]):
        section_list_root = root.find(f".//tei:body", NS)

        body_sections: List[BoxGroup] = []
        body_headings: List[BoxGroup] = []
        body_paragraphs: List[BoxGroup] = []
        body_sentences: List[BoxGroup] = []

        section_divs = section_list_root.findall(f"./tei:div", NS)
        for div in section_divs:
            section_boxes: List[Box] = []
            heading_box_group = self._get_heading_box_group(div)
            if heading_box_group:
                body_headings.append(heading_box_group)
                section_boxes.extend(heading_box_group.boxes)
            for p in div.findall(f"./tei:p", NS):
                paragraph_boxes: List[Box] = []
                paragraph_sentences: List[BoxGroup] = []
                for s in p.findall(f"./tei:s", NS):
                    sentence_boxes = self._xml_coords_to_boxes(s.attrib["coords"])
                    paragraph_sentences.append(BoxGroup(boxes=sentence_boxes))
                    paragraph_boxes.extend(sentence_boxes)
                body_paragraphs.append(BoxGroup(boxes=paragraph_boxes))
                section_boxes.extend(paragraph_boxes)
                body_sentences.extend(paragraph_sentences)
            body_sections.append(BoxGroup(boxes=section_boxes))

        return body_sections, body_headings, body_paragraphs, body_sentences
