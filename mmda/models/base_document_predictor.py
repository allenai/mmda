from dataclasses import dataclass
from abc import abstractmethod
from typing import Union, List, Dict, Any

from mmda.types.annotation import Annotation
from mmda.types.document import Document


class BaseDocumentPredictor:

    ###################################################################
    ##################### Necessary Model Variables ###################
    ###################################################################

    # TODO[Shannon] Add the check for required backends in the future.
    # So different models might require different backends:
    # For example, LayoutLM only needs transformers, but LayoutLMv2
    # needs transformers and Detectron2. It is the model creators'
    # responsibility to check the required backends.
    @property
    @abstractmethod
    def REQUIRED_BACKENDS(self):
        return None

    @property
    @abstractmethod
    def REQUIRED_DOCUMENT_FIELDS(self):
        """Due to the dynamic nature of the document class as well the
        models, we require the model creator to provide a list of required
        fields in the document class. If not None, the predictor class
        will perform the check to ensure that the document contains all
        the specified fields.
        """
        return None

    ###################################################################
    ######################### Core Methods ############################
    ###################################################################

    def __init__(self, model: Any, config: Any):

        self.model = model
        self.config = config

        self.initialize_postprocessor()
        self.initialize_preprocessor()

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

        model_inputs = self.preprocess(document)
        model_outputs = self.model.predict(model_inputs)
        return self.postprocess(model_outputs)

    @abstractmethod
    def initialize_preprocessor(self) -> None:
        """Initialize the necessary preprocessor components"""

    @abstractmethod
    def preprocess(self, document: Document) -> Dict:
        """Convert the input document into the format that is required
        by the model.
        """

    @abstractmethod
    def initialize_postprocessor(self) -> None:
        """Initialize the necessary postprocessor components"""

    @abstractmethod
    def postprocess(self, model_outputs: Any) -> Dict:
        """Convert the model outputs into the Annotation format"""