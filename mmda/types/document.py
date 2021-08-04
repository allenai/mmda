"""



"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Type
import json
import os
from glob import glob

from mmda.types.image import Image
from mmda.types.document_elements import DocumentSymbols
from mmda.types.span import SpanGroup
from mmda.types.annotation import Annotation, DocSpanGroup, Indexer, DocSpanGroupIndexer
from mmda.types.names import Symbols, Images


class Document:

    DEFAULT_FIELDS = [Symbols, Images]
    UNALLOWED_FIELD_NAMES = ['fields']

    def __init__(
        self,
        symbols: DocumentSymbols,
        images: Optional[List["Image.Image"]] = None,
    ):
        self.symbols = symbols
        self.images = images
        self._fields = self.DEFAULT_FIELDS
        self._indexers: Dict[str, Indexer] = {}

    @property
    def fields(self) -> List[str]:
        return self._fields

    def _create_doc_span_group_indexer(self) -> DocSpanGroupIndexer:
        return DocSpanGroupIndexer(num_pages=self.symbols.page_count)

    # TODO: extend implementation to support DocBoxGroup
    def find_overlapping(self, query: Annotation, field_name: str) -> List[Annotation]:
        if not isinstance(query, DocSpanGroup):
            raise NotImplementedError(f'Currently only supports query of type DocSpanGroup')
        return self._indexers[field_name].find(query=query)

    # TODO: this implementation which sets attribute doesn't allow for adding new annos to existing field
    # TODO: extend this to allow fo rother types of groups
    def annotate(self, **kwargs: List[SpanGroup]) -> None:
        """Annotate the fields for document symbols (correlating the annotations with the
        symbols) and store them into the papers.
        """
        # 1) check validity of field names
        for field_name in kwargs.keys():
            assert not field_name.startswith("_"), "The field_name should not start with `_`. "
            assert field_name not in self.fields, "This field name already exists"
            assert field_name not in self.UNALLOWED_FIELD_NAMES, f"The field_name should not be in {self.UNALLOWED_FIELD_NAMES}."
            assert field_name not in dir(self), \
                f"The field_name should not conflict with existing class properties {field_name}"

        # 2) register fields into Document & create span groups
        for field_name, span_groups in kwargs.items():
            self._fields.append(field_name)                                       # save the name of field in doc
            self._annotate_field(span_groups=span_groups, field_name=field_name)  # add span groups to doc + index
            setattr(self, field_name, span_groups)                                # make a property of doc

    def _annotate_field(self, span_groups: List[SpanGroup], field_name: str) -> None:
        """Annotate the Document using a bunch of span groups.
        It will associate the annotations with the document symbols.
        """
        if any([not isinstance(group, SpanGroup) for group in span_groups]):
            raise NotImplementedError(f'Currently doesnt support anything except `SpanGroup` annotation')

        new_doc_span_group_indexer: DocSpanGroupIndexer = self._create_doc_span_group_indexer()
        for span_group in span_groups:
            # 1) create a new DocSpanGroup from SpanGroup
            new_doc_span_group = DocSpanGroup(doc=self, span_group=span_group)

            for span in span_group:
                # 2) Check index if any conflicts (we require disjointness)
                matched_span_group = new_doc_span_group_indexer[span.page_id][span.start:span.end]
                if matched_span_group:
                    raise ValueError(f'Detected overlap with existing SpanGroup {matched_span_group} when attempting index {span}')

                # 3) If no issues, add to index (for each span in span_group)
                new_doc_span_group_indexer[span.page_id][span.start:span.end] = new_doc_span_group

        # add new index to Doc
        self._indexers[field_name] = new_doc_span_group_indexer

    #
    #   to & from JSON
    #

    def to_json(self, fields: Optional[List[str]] = None, with_images=False) -> Dict:
        """Returns a dictionary that's suitable for serialization

        Use `fields` to specify a subset of groups in the Document to include (e.g. 'sentences')
        If `with_images` is True, will also turn the Images into base64 strings.  Else, won't include them.

        Output format looks like
            {
                Symbols: ["...", "...", ...],

            }
        """
        fields = self.fields if fields is None else fields
        if not with_images:
            fields = [field for field in fields if field != Images]
        return {
            field: [group.to_json() for group in getattr(self, field)]
            for field in fields
        }

    @classmethod
    def from_json(cls, doc_dict: Dict):
        fields = doc_dict.keys()        # TODO[kylel]: this modifies the referenced dict, not copy

        # instantiate Document
        symbols = fields.pop(Symbols)
        images = doc_dict.pop(Images, None)
        doc = cls(symbols=symbols, images=images)

        # TODO: unclear if should be `annotations` or `annotation` for `load()`
        for field_name, field_annotations in doc_dict.items():
            field_annotations = [
                DocumentSymbols.load(field_name=field_name, annotations=field_annotation, document=doc)
                for field_annotation in field_annotations
            ]
            doc._add(
                field_name, field_annotations
            )  # We should use add here as they are already annotated

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

        doc_json: Dict = self.to_json(fields=fields, with_images=with_images and images_in_json)

        if with_images and not images_in_json:
            json_path = os.path.join(path, "document.json")     # TODO[kylel]: avoid hard-code

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
                image_files, key=lambda x: int(os.path.basename(x).replace('.png', ''))
            )
            images = [Image.load(image_file) for image_file in image_files]     # TODO[kylel]: not how to load PIL images
        else:
            json_path = path
            images = None

        with open(json_path, "r") as fp:
            json_data = json.load(fp)

        doc = cls.from_json(doc_dict=json_data)
        doc.images = images

        return doc
