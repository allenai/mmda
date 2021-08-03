"""



"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict
import json
import os
from glob import glob

from mmda.types.image import Image
from mmda.types.document_elements import DocumentSymbols, DocumentSpanAnnotationIndexer, DocumentAnnotation, DocSpanGroup


@dataclass
class Document:

    DEFAULT_FIELDS = ["symbols", "images"]

    def __init__(
        self,
        symbols: DocumentSymbols,
        images: Optional["Image.Image"] = None,
    ):

        self.symbols = symbols
        self.images = images
        self._fields = self.DEFAULT_FIELDS
        self._indexers = {}

    @property
    def fields(self):
        return self._fields

    def _check_valid_field_name(self, field_name):
        assert not field_name.startswith(
            "_"
        ), "The field_name should not start with `_`. "
        assert field_name not in ["fields"], "The field_name should not be 'fields'."
        assert field_name not in dir(
            self
        ), f"The field_name should not conflict with existing class properties {field_name}"

    def _register_field(self, field_name):
        if field_name not in self.fields:
            self._check_valid_field_name(field_name)
            self._fields.append(field)
            self._indexers[field_name] = DocumentSpanAnnotationIndexer(num_pages=self.symbols.page_count)

    def _annotate(self, field_name, field_annotations):

        self._register_field(field_name)
        
        for annotation in field_annotations:
            annotation = annotation.annotate(self)
            
        setattr(self, field_name, field_annotations)

    def annotate(self, **annotations: List[DocumentAnnotation]):
        """Annotate the fields for document symbols (correlating the annotations with the
        symbols) and store them into the papers.
        """
        for field_name, field_annotations in annotations.items():
            self._annotate(field_name, field_annotations)

    def _add(self, field_name, field_annotations):

        # This is different from annotate:
        # In add, we assume the annotations are already associated with the symbols
        # and the association is stored in the indexers. As such, we need to ensure 
        # that field and indexers have already been set in some way before calling 
        # this method. I am not totally sure how this mehtod would be used, but it 
        # is a reasonable assumption for now I believe. 
        
        assert field_name in self._fields
        assert field_name in self._indexers

        for annotation in field_annotations:
            assert annotation.doc == self
            # check that the annotation is associated with the document

        setattr(self, field_name, field_annotations)

    def add(self, **annotations: List[DocumentAnnotation]):
        """Add document annotations into this document object.
        Note: in this case, the annotations are assumed to be already associated with
        the document symbols.
        """
        for field_name, field_annotations in annotations.items():
            self._add(field_name, field_annotations)

    def to_json(self, fields: Optional[List[str]] = None, with_images=False):

        fields = self.fields if fields is None else fields
        return {
            field: [ele.to_json() for ele in getattr(self, field)]
            for field in fields
            if field != "images" or with_images
        }

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

        doc_json = self.to_json(fields, with_images=with_images and images_in_json)

        if with_images and not images_in_json:
            json_path = os.path.join(path, "document.json")

            with open(json_path, "w") as fp:
                json.dump(doc_json, fp)

            for pid, image in enumerate(self.images):
                image.save(os.path.join(path, f"{pid}.png"))
        else:
            with open(path, "w") as fp:
                json.dump(doc_json, fp)

    @classmethod
    def from_json(cls, json_data: Dict):
        fields = json_data.keys()

        symbols = fields.pop("symbols")  # TODO: avoid hard-coded values
        images = json_data.pop("images", None)
        doc = cls(symbols, images)

        for field_name, field_annotations in json_data.items():
            field_annotations = [
                load_document_element(field_name, field_annotation, document=doc)
                for field_annotation in field_annotations
            ]
            doc._add(
                field_name, field_annotations
            )  # We should use add here as they are already annotated

        return doc

    @classmethod
    def load(cls, path: str):

        if os.path.isdir(path):
            json_path = os.path.join(path, "document.json")
            image_files = glob(os.path.join(path, "*.png"))
            image_files = sorted(
                image_files, key=lambda x: int(os.path.basename(x)[:-4])
            )
            images = [Image.load(image_file) for image_file in image_files]
        else:
            json_path = path
            images = None

        with open(json_path, "r") as fp:
            json_data = json.load(fp)

        doc = cls.from_json(json_data)
        doc.images = images

        return doc

    @classmethod
    def find(self, query: DocSpanGroup, field_name: str):
        
        # As for now query only supports for DocSpanGroup, the function is 
        # just this simple
        
        return self._indexers[field_name].index(query)