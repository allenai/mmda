"""

Dataclass for creating token streams from a document via SymbolScraper

@kylel

"""

from typing import Optional, List, Dict, Tuple

import os
import json
import subprocess

import re
from collections import defaultdict

from mmda.types.span import Span
from mmda.types.document import Document
from mmda.types.boundingbox import BoundingBox
from mmda.parsers.parser import Parser


class SymbolScraperParser(Parser):
    def __init__(self, sscraper_bin_path: str):
        self.sscraper_bin_path = sscraper_bin_path

    def parse(self, infile: str, outdir: str) -> str:
        xmlfile = self._run_sscraper(infile=infile, outdir=outdir)
        sscraper_json = self._parse_sscraper_xml(xmlfile=xmlfile)
        outfile = os.path.join(outdir, infile.replace('.pdf', '.json'))
        with open(outfile, 'w') as f_out:
            json.dump(f_out, sscraper_json, indent=4)
        return outfile

    def load(self, infile: str) -> Document:
        with open(infile) as f_in:
            doc_json = json.load(f_in)
            doc = Document.from_json(doc_json=doc_json)
            return doc

    #
    #   methods for interacting with SymbolScraper binary
    #
    def _run_sscraper(self, infile: str, outdir: str) -> str:
        """Returns xmlpath of parsed output"""
        if not infile.endswith('.pdf'):
            raise FileNotFoundError(f'{infile} doesnt end in .pdf extension, which {self} expected')
        os.makedirs(outdir, exist_ok=True)
        cmd = [self.sscraper_bin_path, '-b', infile, outdir]
        subprocess.run(cmd)
        xmlfile = os.path.join(outdir, infile.replace('.pdf', '.xml'))
        if not os.path.exists(xmlfile):
            raise FileNotFoundError(f'Parsing {infile} may have failed. Cant find {xmlfile}.')
        else:
            return xmlfile

    #
    #   methods for building bbox from sscraper XML
    #
    def _build_from_sscraper_bbox(self, sscraper_bbox: str,
                                  sscraper_page_width: float,
                                  sscraper_page_height: float,
                                  page_id: int) -> BoundingBox:
        left, top, width, height = [float(element) for element in sscraper_bbox.split(' ')]
        return BoundingBox(l=left / sscraper_page_width,
                           t=(sscraper_page_height - top) / sscraper_page_height,
                           # annoyingly, sscraper goes other way
                           w=width / sscraper_page_width,
                           h=height / sscraper_page_height,
                           page=page_id)

    #
    #   helper methods for parsing sscraper's particular XML format
    #
    def _split_list_by_start_end_tags(self, my_list: List[str],
                                      start_tag: str,
                                      end_tag: str) -> List[Tuple[int, int]]:
        """Basically hunts through `my_list` for start & end tag values and returns (start, end) indices"""
        start = None
        end = None
        current = 0
        my_start_ends: List[Tuple[int, int]] = []
        while current < len(my_list):
            if start_tag in my_list[current]:
                start = current
            elif end_tag in my_list[current]:
                end = current + 1
            else:
                pass
            if start is not None and end is not None:
                my_start_ends.append((start, end))
                start = None
                end = None
            current += 1
        return my_start_ends

    def _find_one_and_extract(self, my_list: List[str],
                              start_tag: str,
                              end_tag: str) -> Optional[str]:
        """Hunts for an element in `my_list` to be bounded within start & end tag values."""
        for element in my_list:
            if start_tag in element and end_tag in element:
                return element.replace(start_tag, '').replace(end_tag, '')
        return None

    def _parse_row_head_tag(self, row_tag: str) -> Dict:
        # TODO - not sure why line bboxes are useful; skip for now.  they dont quite make sense (e.g. bbox[1] == bbox[3])
        match = re.match(pattern=r'<Line id=\"([0-9]+)\" BBOX=\"(.+)\">', string=row_tag)
        return {'id': int(match.group(1)), 'bbox': match.group(2)}

    def _parse_word_head_tag(self, word_tag: str) -> Dict:
        match = re.match(pattern=r'<Word id=\"([0-9]+)\">', string=word_tag)
        return {'id': int(match.group(1))}

    def _parse_char_head_tag(self, char_tag: str) -> Dict:
        match = re.match(pattern=r'<Char id=\"([0-9]+)\" mergeId=\"(.+)\" BBOX=\"(.+)\" (.*?)>(.+)</Char>',
                         string=char_tag)  # RGB string is inconsistent formatting so skip
        return {'id': match.group(1), 'bbox': match.group(3), 'text': match.group(5)}

    #
    #   main parsing method
    #
    def _parse_sscraper_xml(self, xmlfile: str) -> Dict:

        with open(xmlfile, 'r') as f_in:
            xml_lines = [line.strip() for line in f_in]

        # get runtime
        runtime = int(self._find_one_and_extract(my_list=xml_lines, start_tag='<runtime>', end_tag='</runtime>'))
        if runtime is None:
            raise ValueError(f'No Runtime for {xmlfile}')

        # get page metrics
        start, end = self._split_list_by_start_end_tags(my_list=xml_lines,
                                                        start_tag='<pagemetrics>',
                                                        end_tag='</pagemetrics>')[0]  # just one of them exists
        pagemetrics = xml_lines[start:end]

        page_to_metrics = {}
        for start, end in self._split_list_by_start_end_tags(my_list=pagemetrics,
                                                             start_tag='<page>',
                                                             end_tag='</page>'):
            partition = pagemetrics[start:end]
            page_num = int(self._find_one_and_extract(my_list=partition, start_tag='<no>', end_tag='</no>'))
            page_width = float(
                self._find_one_and_extract(my_list=partition, start_tag='<pagewidth>', end_tag='</pagewidth>'))
            page_height = float(
                self._find_one_and_extract(my_list=partition, start_tag='<pageheight>', end_tag='</pageheight>'))
            page_num_lines = int(
                self._find_one_and_extract(my_list=partition, start_tag='<lines>', end_tag='</lines>'))
            page_num_words = int(
                self._find_one_and_extract(my_list=partition, start_tag='<words>', end_tag='</words>'))
            page_num_chars = int(
                self._find_one_and_extract(my_list=partition, start_tag='<characters>', end_tag='</characters>'))
            page_to_metrics[page_num] = {
                'height': page_height,
                'width': page_width,
                'lines': page_num_lines,
                'words': page_num_words,
                'chars': page_num_chars
            }

        # get token stream (grouped by page & row)
        page_to_row_to_words = defaultdict(lambda: defaultdict(list))
        for page_start, page_end in self._split_list_by_start_end_tags(my_list=xml_lines,
                                                                       start_tag='<Page',
                                                                       end_tag='</Page>'):
            page_lines = xml_lines[page_start:page_end]
            page_id = int(page_lines[0].replace('<Page id="', '').replace('">', ''))
            for row_start, row_end in self._split_list_by_start_end_tags(my_list=page_lines,
                                                                         start_tag='<Line',
                                                                         end_tag='</Line>'):
                row_lines = page_lines[row_start:row_end]
                row_info = self._parse_row_head_tag(row_tag=row_lines[0])  # first line is the head tag
                row_id = row_info['id']
                for word_start, word_end in self._split_list_by_start_end_tags(my_list=row_lines,
                                                                               start_tag='<Word',
                                                                               end_tag='</Word>'):
                    word_lines = row_lines[word_start:word_end]
                    word_info = self._parse_word_head_tag(word_tag=word_lines[0])  # first line is the head tag
                    char_bboxes: List[BoundingBox] = []
                    word = ''
                    for char_tag in [w for w in word_lines if w.startswith('<Char') and w.endswith('</Char>')]:
                        char_info = self._parse_char_head_tag(char_tag=char_tag)
                        bbox = self._build_from_sscraper_bbox(sscraper_bbox=char_info['bbox'],
                                                              sscraper_page_width=page_to_metrics[page_id]['width'],
                                                              sscraper_page_height=page_to_metrics[page_id]['height'],
                                                              page_id=page_id)
                        char_bboxes.append(bbox)
                        word += char_info['text']
                    # sometimes, just empty words (e.g. figures)
                    if not word or not char_bboxes:
                        continue
                    else:
                        word_bbox = BoundingBox.union_bboxes(bboxes=char_bboxes)
                        page_to_row_to_words[page_id][row_id].append(Token(text=word, bbox=word_bbox))

        return {
            'runtime': runtime,
            'page_to_metrics': page_to_metrics,
            'page_to_row_to_words': {
                page: {row: words for row, words in row_to_words.items()} for
                page, row_to_words in page_to_row_to_words.items()
            }
        }


