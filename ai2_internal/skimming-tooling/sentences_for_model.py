from enum import Enum
from pathlib import Path
import re
import warnings
from typing import Dict, List, NamedTuple, Sequence, Union

import torch
from cached_path import cached_path

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import \
    DictionaryWordPredictor
from mmda.predictors.heuristic_predictors.sentence_boundary_predictor import \
    PysbdSentenceBoundaryPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer
from mmda.types.annotation import BoxGroup, SpanGroup
from mmda.types.document import Document
from mmda.types.names import Words, Sentences, Blocks

import layoutparser as lp

WORDS_URL = 'https://github.com/dwyl/english-words/raw/master/words.txt'


def draw_blocks(
    image,
    doc_tokens: List[SpanGroup],
    color_map=None,
    token_boundary_width=0,
    alpha=0.25,
    **kwargs,
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            lp.Rectangle(
                *token.box_group.boxes[0]
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=token.type or token.box_group.type,
            text=token.symbols[0],
        )
        for token in doc_tokens
    ]
    return lp.draw_box(
        image,
        layout,
        color_map=color_map,
        box_color='grey' if not color_map else None,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **kwargs,
    )



class ExtendedDictionaryWordPredictor(DictionaryWordPredictor):

    def run_small_caps_heuristic(
        self: 'ExtendedDictionaryWordPredictor',
        lead: SpanGroup,
        tail: SpanGroup,
    ) -> bool:
        assert lead.text is not None and tail.text is not None, \
            "SpanGroup must have text."

        # check if both spans contain capital letters
        is_both_all_caps = (
            re.match(r'^[\-\(\[]{0,1}[A-Z\-]+$', lead.text)
            is not None and
            re.match(r'^[A-Z\-]+[\)\]]{0,1}[\-,\.\?\!\;\:]{0,1}$', tail.text)
            is not None
        )

        # if this is a word with small caps, then the last span for the first
        # group and the first span for the second group should have different
        # letter heights

        is_different_heights = (
            lead.spans[-1].box.h != tail.spans[0].box.h  # type: ignore
        )

        return is_both_all_caps and is_different_heights

    def predict(self: 'ExtendedDictionaryWordPredictor',
                document: Document) -> List[SpanGroup]:
        """Get words from a document as a list of SpanGroup.
        """

        words = super().predict(document)

        i = 0  # count explicitly because length of `words` is changing
        while i < (len(words) - 1):
            to_merge = self.run_small_caps_heuristic(lead := words[i],
                                                     tail := words[i + 1])

            if to_merge:
                # spans are simply concatenated
                new_spans = lead.spans + tail.spans

                # bit of logic to determine if any of the spans to merge have
                # attribute box_group set to not None, and if so, deal with
                # merging them properly.
                if lead_bg := lead.box_group:
                    if tail_bg := tail.box_group:
                        new_box_groups = BoxGroup(
                            boxes=(lead_bg.boxes + tail_bg.boxes),
                            type=(lead_bg.type or tail_bg.type)
                        )
                    else:
                        new_box_groups = lead.box_group
                elif tail.box_group:
                    new_box_groups = tail.box_group
                else:
                    new_box_groups = None

                # the new text for the merge span group is the concatenation
                # of the text of the two spans if at least one has text,
                # otherwise it is None
                new_text = ((lead.text or '') + (tail.text or '') or None)

                # we give lead token precedence over tail token in type
                new_type = (lead.type or tail.type)

                # make new span group, replace the first of the two existing
                # ones, then toss away the second one.
                merged = SpanGroup(spans=new_spans,
                                   id=i,
                                   text=new_text,
                                   type=new_type,
                                   box_group=new_box_groups)
                words[i] = merged
                words.pop(i + 1)

            else:
                i += 1
                # refresh the word id bc list will (potentially) get shorter
                # as we merge
                words[i].id = i

        return words

def span_is_fully_contained(container: SpanGroup,
                            maybe_contained: SpanGroup) -> bool:
    return all(
        any(container_span.start <= maybe_contained_span.start
            and container_span.end >= maybe_contained_span.end
            for container_span in container.spans)
        for maybe_contained_span in maybe_contained.spans
    )


class SectionTypes:
    text: str = 'Text'
    title: str = 'Title'
    list: str = 'List'
    table: str = 'Table'
    figure: str = 'Figure'
    other: str = 'Other'
    ref_app: str = 'ReferencesAppendix'
    abstract: str = 'Abstract'

    @classmethod
    def make_color_map(cls) -> Dict[str, str]:
        return {cls.title: 'red',
                cls.text: 'blue',
                cls.figure: 'green',
                cls.table: 'yellow',
                cls.other: 'grey',
                cls.ref_app: 'purple',
                cls.abstract: 'magenta'}


def tag_abstract_blocks(doc: Document):
    has_seen_abstract = False
    in_abstract = False
    for block in doc.blocks:
        if block.box_group.type == SectionTypes.title:
            sec_type = re.sub(
                # remove leading section numbers if present
                r'^(\d|[\.])+\s+',
                '',
                ' '.join(w.text for w in block.words)
            ).lower()
            if has_seen_abstract:
                break
            # HEURISTIC only check for match in the first 20 chars or so
            if 'abstract' in sec_type[:20]:
                has_seen_abstract = True
                in_abstract = True
        if in_abstract:
            block.type = SectionTypes.abstract


def tag_references_blocks(doc: Document):
    in_references = False
    for block in doc.blocks:
        if block.box_group.type == SectionTypes.title:
            sec_type = re.sub(
                # remove leading section numbers if present
                r'^(\d|[\.])+\s+',
                '',
                ' '.join(w.text for w in block.words)
            ).lower()
            # HEURISTIC only check for match in the first 20 chars or so
            if 'references' in sec_type[:20]:
                in_references = True
        if in_references:
            block.type = SectionTypes.ref_app


def tag_text_blocks(doc: Document):
    for block in doc.blocks:
        if (
            block.box_group.type == SectionTypes.text or
            block.box_group.type == SectionTypes.list
        ):
            block_type = SectionTypes.text
        elif block.box_group.type == SectionTypes.title:
            sents = [sent for sent in block.sents if
                     span_is_fully_contained(block, sent)]
            if len(sents) >= 2:
                # heuristic: something tagged as a title with at
                # least two fully contained sentences is probably a text
                block_type = SectionTypes.text
            else:
                block_type = SectionTypes.title
        else:
            block_type = block.box_group.type or SectionTypes.other
        block.type = block_type



def main(path: Union[str, Path]):
    path = Path(path)

    pdfplumber_parser = PDFPlumberParser()
    rasterizer = PDF2ImageRasterizer()
    layout_predictor = LayoutParserPredictor.from_pretrained(
        "lp://efficientdet/PubLayNet"
    )
    word_predictor = ExtendedDictionaryWordPredictor(
        str(cached_path(WORDS_URL))
    )
    sentence_predictor = PysbdSentenceBoundaryPredictor()

    doc = pdfplumber_parser.parse(str(path))
    images = rasterizer.rasterize(str(path), dpi=72)
    doc.annotate_images(images)

    with torch.no_grad(), warnings.catch_warnings():
        layout_regions = layout_predictor.predict(doc)
        doc.annotate(blocks=layout_regions)

    words = word_predictor.predict(doc)
    doc.annotate(words=words)

    sents = sentence_predictor.predict(doc)
    doc.annotate(sents=sents)

    tag_text_blocks(doc)
    tag_abstract_blocks(doc)
    tag_references_blocks(doc)

    for pid in range(len(doc.pages)):
        viz = draw_blocks(doc.images[pid],
                          doc.pages[pid].blocks,
                          color_map=SectionTypes.make_color_map(),
                          alpha=0.3)
        path.with_suffix("").mkdir(parents=True, exist_ok=True)
        viz.save(path.with_suffix("") / f"{pid}.png")

    # for block in doc.cleaned_blocks:
    #     print('==========================')
    #     # print(f'BLOCK ({block.type}): ' + ' '.join(w.text for w in block.words))
    #     for sent in block.sents:
    #         if not span_is_fully_contained(block, sent):
    #             print(f'IGNORE ({block.type}): ' + ' '.join(w.text for w in sent.words))
    #         else:
    #             print(f'SENTENCE ({block.type}): ' + ' '.join(w.text for w in sent.words))
    #     print('==========================')



if __name__ == '__main__':
    path = '/Users/lucas/Downloads/test_pdfs_block/test3.pdf'
    main(path)
