"""



"""

import itertools
import json
import os
import warnings
from copy import deepcopy
from glob import glob
from typing import Dict, Iterable, List, Optional

from mmda.types.annotation import Annotation, BoxGroup, SpanGroup
from mmda.types.image import PILImage
from mmda.types.indexers import Indexer, SpanGroupIndexer
from mmda.types.names import Images, Symbols
from mmda.utils.tools import allocate_overlapping_tokens_for_box, merge_neighbor_spans


class Document:

    SPECIAL_FIELDS = [Symbols, Images]
    UNALLOWED_FIELD_NAMES = ["fields"]

    def __init__(self, symbols: str):
        self.symbols = symbols
        self.images = []
        self.__fields = []
        self.__indexers: Dict[str, Indexer] = {}

    @property
    def fields(self) -> List[str]:
        return self.__fields

    # TODO: extend implementation to support DocBoxGroup
    def find_overlapping(self, query: Annotation, field_name: str) -> List[Annotation]:
        if not isinstance(query, SpanGroup):
            raise NotImplementedError(
                f"Currently only supports query of type SpanGroup"
            )
        return self.__indexers[field_name].find(query=query)

    def annotate(
        self, is_overwrite: bool = False, **kwargs: Iterable[Annotation]
    ) -> None:
        """Annotate the fields for document symbols (correlating the annotations with the
        symbols) and store them into the papers.
        """
        # 1) check validity of field names
        for field_name in kwargs.keys():
            assert (
                field_name not in self.SPECIAL_FIELDS
            ), f"The field_name {field_name} should not be in {self.SPECIAL_FIELDS}."

            assert field_name not in dir(
                self
            ), f"The field_name {field_name} should not conflict with existing class properties"

            if field_name in self.fields and not is_overwrite:
                raise AssertionError(
                    f"This field name {field_name} already exists. To override, set `is_overwrite=True`"
                )

        # 2) register fields into Document
        for field_name, annotations in kwargs.items():
            if len(annotations) == 0:
                warnings.warn(f"The annotations is empty for the field {field_name}")
                continue

            # TODO: why is this good idea?
            annotations = deepcopy(annotations)

            annotation_types = {type(a) for a in annotations}
            assert (
                len(annotation_types) == 1
            ), f"Annotations in field_name {field_name} more than 1 type: {annotation_types}"
            annotation_type = annotation_types.pop()

            if annotation_type == SpanGroup:
                span_groups = self._annotate_span_group(
                    span_groups=annotations, field_name=field_name
                )
            elif annotation_type == BoxGroup:
                # TODO: not good. BoxGroups should be stored on their own, not auto-generating SpanGroups.
                span_groups = self._annotate_box_group(
                    box_groups=annotations, field_name=field_name
                )
            else:
                raise NotImplementedError(
                    f"Unsupported annotation type {annotation_type} for {field_name}"
                )

            # register fields
            setattr(self, field_name, span_groups)
            self.__fields.append(field_name)

    def annotate_images(
        self, images: Iterable[PILImage], is_overwrite: bool = False
    ) -> None:
        if not is_overwrite and len(self.images) > 0:
            raise AssertionError(
                "This field name {Images} already exists. To override, set `is_overwrite=True`"
            )

        if len(images) == 0:
            raise AssertionError("No images were provided")

        image_types = {type(a) for a in images}
        assert len(image_types) == 1, f"Images contain more than 1 type: {image_types}"
        image_type = image_types.pop()

        if not issubclass(image_type, PILImage):
            raise NotImplementedError(
                f"Unsupported image type {image_type} for {Images}"
            )

        self.images = images

    def _annotate_span_group(
        self, span_groups: List[SpanGroup], field_name: str
    ) -> List[SpanGroup]:
        """Annotate the Document using a bunch of span groups.
        It will associate the annotations with the document symbols.
        """
        assert all([isinstance(group, SpanGroup) for group in span_groups])

        new_span_group_indexer = SpanGroupIndexer()
        for span_group in span_groups:

            # 1) add Document to each SpanGroup
            span_group.attach_doc(doc=self)

            # 2) for each span in span_group, (a) check if any conflicts (requires disjointedness).
            #    then (b) add span group to index at this span location
            #  TODO: can be cleaned up; encapsulate the checker somewhere else
            for span in span_group:
                # a) Check index if any conflicts (we require disjointness)
                matched_span_group = new_span_group_indexer[span.start : span.end]
                if matched_span_group:
                    raise ValueError(
                        f"Detected overlap with existing SpanGroup {matched_span_group} when attempting index {span_group}"
                    )

                # b) If no issues, add to index (for each span in span_group)
                new_span_group_indexer[span.start : span.end] = span_group

        # add new index to Doc
        self.__indexers[field_name] = new_span_group_indexer

        return span_groups

    def _annotate_box_group(
        self, box_groups: List[BoxGroup], field_name: str
    ) -> List[SpanGroup]:
        """Annotate the Document using a bunch of box groups.
        It will associate the annotations with the document symbols.
        """
        assert all([isinstance(group, BoxGroup) for group in box_groups])

        all_page_tokens = dict()
        derived_span_groups = []

        for box_id, box_group in enumerate(box_groups):

            all_token_spans_with_box_group = []

            for box in box_group.boxes:

                # Caching the page tokens to avoid duplicated search

                if box.page not in all_page_tokens:
                    cur_page_tokens = all_page_tokens[box.page] = list(
                        itertools.chain.from_iterable(
                            span_group.spans
                            for span_group in self.pages[box.page].tokens
                        )
                    )
                else:
                    cur_page_tokens = all_page_tokens[box.page]

                # Find all the tokens within the box
                tokens_in_box, remaining_tokens = allocate_overlapping_tokens_for_box(
                    token_spans=cur_page_tokens, box=box
                )
                all_page_tokens[box.page] = remaining_tokens

                all_token_spans_with_box_group.extend(tokens_in_box)

            derived_span_groups.append(
                SpanGroup(
                    spans=merge_neighbor_spans(
                        spans=all_token_spans_with_box_group, distance=1
                    ),
                    box_group=box_group,
                    # id = box_id,
                )
                # TODO Right now we cannot assign the box id, or otherwise running doc.blocks will
                # generate blocks out-of-the-specified order.
            )

        del all_page_tokens

        derived_span_groups = sorted(
            derived_span_groups, key=lambda span_group: span_group.start
        )
        # ensure they are ordered based on span indices

        for box_id, span_group in enumerate(derived_span_groups):
            span_group.id = box_id

        return self._annotate_span_group(
            span_groups=derived_span_groups, field_name=field_name
        )

    #
    #   to & from JSON
    #

    def to_json(self, fields: Optional[List[str]] = None, with_images=False) -> Dict:
        """Returns a dictionary that's suitable for serialization

        Use `fields` to specify a subset of groups in the Document to include (e.g. 'sentences')
        If `with_images` is True, will also turn the Images into base64 strings.  Else, won't include them.

        Output format looks like
            {
                symbols: "...",
                field1: [...],
                field2: [...]
            }
        """
        doc_dict = {Symbols: self.symbols}
        if with_images:
            doc_dict[Images] = [image.to_json() for image in self.images]

        # figure out which fields to serialize
        fields = (
            self.fields if fields is None else fields
        )  # use all fields unless overridden

        # add to doc dict
        for field in fields:
            doc_dict[field] = [
                doc_span_group.to_json() for doc_span_group in getattr(self, field)
            ]

        return doc_dict

    @classmethod
    def from_json(cls, doc_dict: Dict) -> "Document":
        # 1) instantiate basic Document
        symbols = doc_dict[Symbols]
        doc = cls(symbols=symbols)

        images_dict = doc_dict.get(Images, None)
        if images_dict:
            doc.annotate_images(
                [PILImage.frombase64(image_str) for image_str in images_dict]
            )

        # 2) convert span group dicts to span gropus
        field_name_to_span_groups = {}
        for field_name, span_group_dicts in doc_dict.items():
            if field_name not in doc.SPECIAL_FIELDS:
                span_groups = [
                    SpanGroup.from_json(span_group_dict=span_group_dict)
                    for span_group_dict in span_group_dicts
                ]
                field_name_to_span_groups[field_name] = span_groups

        # 3) load annotations for each field
        doc.annotate(**field_name_to_span_groups)

        return doc

    #
    #   for serialization
    #

    def save(
        self,
        path: str,
        fields: Optional[List[str]] = None,
        with_images=True,
        images_in_json=False,
    ):

        if with_images and not images_in_json:
            assert os.path.isdir(
                path
            ), f"When with_images={with_images} and images_in_json={images_in_json}, it requires the path to be a folder"
            # f-string equals like f"{with_images=}" will break the black formatter and won't work for python < 3.8

        doc_json: Dict = self.to_json(
            fields=fields, with_images=with_images and images_in_json
        )

        if with_images and not images_in_json:
            json_path = os.path.join(
                path, "document.json"
            )  # TODO[kylel]: avoid hard-code

            with open(json_path, "w") as fp:
                json.dump(doc_json, fp)

            for pid, image in enumerate(self.images):
                image.save(os.path.join(path, f"{pid}.png"))
        else:
            with open(path, "w") as fp:
                json.dump(doc_json, fp)

    @classmethod
    def load(cls, path: str) -> "Document":
        """Instantiate a Document object from its serialization.
        If path is a directory, loads the JSON for the Document along with all Page images
        If path is a file, just loads the JSON for the Document, assuming no Page images"""
        if os.path.isdir(path):
            json_path = os.path.join(path, "document.json")
            image_files = glob(os.path.join(path, "*.png"))
            image_files = sorted(
                image_files, key=lambda x: int(os.path.basename(x).replace(".png", ""))
            )
            images = [PILImage.load(image_file) for image_file in image_files]
        else:
            json_path = path
            images = None

        with open(json_path, "r") as fp:
            json_data = json.load(fp)

        doc: Document = cls.from_json(doc_dict=json_data)
        doc.images = images

        return doc
