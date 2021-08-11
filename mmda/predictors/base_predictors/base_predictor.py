from dataclasses import dataclass
from abc import abstractmethod
from typing import Union, List, Dict, Any

from mmda.types.annotation import Annotation
from mmda.types.document import Document


class BasePredictor:

    ###################################################################
    ##################### Necessary Model Variables ###################
    ###################################################################

    REQUIRED_BACKENDS: List[str] = []
    REQUIRED_DOCUMENT_FIELDS: List[str] = []

    ###################################################################
    ######################### Core Methods ############################
    ###################################################################

    def _doc_field_checker(self, document: Document) -> None:
        if self.REQUIRED_DOCUMENT_FIELDS is not None:
            for field in self.REQUIRED_DOCUMENT_FIELDS:
                assert (
                    field in document.fields
                ), f"The input Document object {document} doesn't contain the required field {field}"

    # TODO[Shannon] Allow for some preprocessed document intput
    # representation for better performance?
    @abstractmethod
    def predict(self, document: Document) -> List[Annotation]:
        """For all the mmda models, the input is a document object, and
        the output is a list of annotations.
        """
        self._doc_field_checker(document)