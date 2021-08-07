"""

Dataclass for creating token streams from a document via SymbolScraper

@kylel

"""

from typing import Optional, List, Dict, Tuple, Union

import os
import json
import subprocess
import tempfile

import re
from collections import defaultdict

from mmda.types.box import Box
from mmda.types.document import Document
from mmda.parsers.parser import BaseParser


class SymbolScraperParser(BaseParser):
    def __init__(self, sscraper_bin_path: str):
        self.sscraper_bin_path = sscraper_bin_path

    def parse(self, input_pdf_path: str, output_json_path: Optional[str] = None,
              tempdir: Optional[str] = None, load_images=False) -> Document:

        if tempdir is None:
            with tempfile.TemporaryDirectory() as tempdir:
                xmlfile = self._run_sscraper(input_pdf_path=input_pdf_path, outdir=tempdir)
                doc: Document = self._parse_xml_to_doc(xmlfile=xmlfile)
        else:
            xmlfile = self._run_sscraper(input_pdf_path=input_pdf_path, outdir=tempdir)
            doc: Document = self._parse_xml_to_doc(xmlfile=xmlfile)

        if load_images:
            raise NotImplementedError(f'Load images?')

        if output_json_path:
            with open(output_json_path, 'w') as f_out:
                json.dump(doc.to_json(), f_out, indent=4)

        return doc

    #
    #   methods for interacting with SymbolScraper binary
    #
    def _run_sscraper(self, input_pdf_path: str, outdir: str) -> str:
        """Returns xmlpath of parsed output"""
        if not input_pdf_path.endswith('.pdf'):
            raise FileNotFoundError(f'{input_pdf_path} doesnt end in .pdf extension, which {self} expected')
        os.makedirs(outdir, exist_ok=True)
        cmd = [self.sscraper_bin_path, '-b', input_pdf_path, outdir]
        subprocess.run(cmd)
        xmlfile = os.path.join(outdir, os.path.basename(input_pdf_path).replace('.pdf', '.xml'))
        if not os.path.exists(xmlfile):
            raise FileNotFoundError(f'Parsing {input_pdf_path} may have failed. Cant find {xmlfile}.')
        else:
            return xmlfile

    #
    #   methods for building bbox from sscraper XML
    #
    def _build_from_sscraper_bbox(self, sscraper_bbox: str,
                                  sscraper_page_width: float,
                                  sscraper_page_height: float,
                                  page_id: int) -> Box:
        left, top, width, height = [float(element) for element in sscraper_bbox.split(' ')]
        return Box(l=left / sscraper_page_width,
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
    #   main parsing methods
    #
    def _parse_page_to_metrics(self, xml_lines: List[str]) -> Dict:
        start, end = self._split_list_by_start_end_tags(my_list=xml_lines,
                                                        start_tag='<pagemetrics>',
                                                        end_tag='</pagemetrics>')[0]  # just one of them exists
        pagemetrics = xml_lines[start:end]

        page_to_metrics = {}
        for start, end in self._split_list_by_start_end_tags(my_list=pagemetrics, start_tag='<page>', end_tag='</page>'):
            partition = pagemetrics[start:end]
            page_num = int(self._find_one_and_extract(my_list=partition, start_tag='<no>', end_tag='</no>'))
            page_width = float(self._find_one_and_extract(my_list=partition, start_tag='<pagewidth>', end_tag='</pagewidth>'))
            page_height = float(self._find_one_and_extract(my_list=partition, start_tag='<pageheight>', end_tag='</pageheight>'))
            page_num_rows = int(self._find_one_and_extract(my_list=partition, start_tag='<lines>', end_tag='</lines>'))
            page_num_words = int(self._find_one_and_extract(my_list=partition, start_tag='<words>', end_tag='</words>'))
            page_num_chars = int(self._find_one_and_extract(my_list=partition, start_tag='<characters>', end_tag='</characters>'))
            page_to_metrics[page_num] = {
                'height': page_height,
                'width': page_width,
                'rows': page_num_rows,
                'words': page_num_words,
                'chars': page_num_chars
            }
        return page_to_metrics

    def _parse_page_to_row_to_words(self, xml_lines: List[str], page_to_metrics: Dict) -> Dict:
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
                        page_to_row_to_words[page_id][row_id].append({'text': word, 'bbox': word_bbox})
        return {
            page: {row: words for row, words in row_to_words.items()}
            for page, row_to_words in page_to_row_to_words.items()
        }

    def _convert_nested_text_to_doc_json(self, page_to_row_to_words: Dict) -> Dict:
        text = ''
        pages: List[Span] = []
        tokens: List[Span] = []
        rows: List[Span] = []
        start = 0
        for i, (page, row_to_words) in enumerate(page_to_row_to_words.items()):
            page_rows = []
            for j, (row, words) in enumerate(row_to_words.items()):
                # process tokens in this row
                row_tokens = []
                for k, word in enumerate(words):
                    text += word['text']
                    end = start + len(word['text'])
                    # make token
                    token = Span(start=start, end=end, id=len(tokens), bbox=word['bbox'])
                    row_tokens.append(token)
                    tokens.append(token)
                    if k < len(words) - 1:
                        text += ' '
                    else:
                        text += '\n'
                    start = end + 1
                # make row
                row = Span(start=row_tokens[0].start, end=row_tokens[-1].end, id=len(rows),
                           bbox=BoundingBox.union_bboxes(bboxes=[token.bbox for token in row_tokens]))
                page_rows.append(row)
                rows.append(row)
            # make page
            page = Span(start=page_rows[0].start, end=page_rows[-1].end, id=i,
                        bbox=BoundingBox.union_bboxes(bboxes=[row.bbox for row in page_rows]))
            pages.append(page)
        return {
            Text: text,
            Page: [page.to_json() for page in pages],
            Token: [token.to_json() for token in tokens],
            Row: [row.to_json() for row in rows],
            Sent: [],
            Block: []
        }

    def _parse_xml_to_doc(self, xmlfile: str) -> Document:

        with open(xmlfile, 'r') as f_in:
            xml_lines = [line.strip() for line in f_in]

        # get runtime
        runtime = int(self._find_one_and_extract(my_list=xml_lines, start_tag='<runtime>', end_tag='</runtime>'))
        if runtime is None:
            raise ValueError(f'No Runtime for {xmlfile}')
        else:
            print(f'Symbol Scraper took {runtime} sec for {xmlfile}...')

        # get page metrics
        page_to_metrics = self._parse_page_to_metrics(xml_lines=xml_lines)
        print(f'\tNum pages: {len(page_to_metrics)}')
        print(f"\tAvg words: {sum([metric['words'] for metric in page_to_metrics.values()]) / len(page_to_metrics)}")
        print(f"\tAvg rows: {sum([metric['rows'] for metric in page_to_metrics.values()]) / len(page_to_metrics)}")

        # get token stream (grouped by page & row)
        page_to_row_to_words = self._parse_page_to_row_to_words(xml_lines=xml_lines, page_to_metrics=page_to_metrics)

        # convert to spans
        doc_json = self._convert_nested_text_to_doc_json(page_to_row_to_words=page_to_row_to_words)

        # build Document
        doc = Document.from_json(doc_json=doc_json)
        return doc


