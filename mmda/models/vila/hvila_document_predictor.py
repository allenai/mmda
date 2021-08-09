from typing import Union, List, Dict, Any
from transformers import AutoTokenizer

from mmda.models.base_document_predictor import BaseDocumentPredictor
from mmda.models.base_predictor_config import BaseDocumentPredictorConfig
from mmda.types.names import *
from mmda.types.document import Document

from mmda.models.vila.modeling import (
    HierarchicalModelForTokenClassification,
    HierarchicalModelConfig,
)
from mmda.models.vila.preprocessors import instantiate_vila_model_preprocessor


class HvilaDocumentPredictor(BaseDocumentPredictor):

    REQUIRED_BACKENDS = ["torch", "transformers"]
    DEFAULT_DOCUMENT_FIELDS = [Tokens, Pages]

    def __init__(self, model: Any, config: Any, tokenizer: Any):

        super().__init__(model, config)
        self.tokenizer = tokenizer

        # REQUIRED_DOCUMENT_FIELDS is not longer a class property,
        # but a object-level property, as it will be dynamically
        # determined based on the config
        if self.config.agg_level == Blocks:
            self.REQUIRED_DOCUMENT_FIELDS = self.DEFAULT_DOCUMENT_FIELDS + [Blocks]
        elif self.config.agg_level == Rows:
            self.REQUIRED_DOCUMENT_FIELDS = self.DEFAULT_DOCUMENT_FIELDS + [Rows]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs):

        model = HierarchicalModelForTokenClassification.from_pretrained(
            model_name_or_path
        )
        config = BaseDocumentPredictorConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, config, tokenizer, *args, **kwargs)

    def initialize_preprocessor(self):

        self.preprocessor = instantiate_vila_model_preprocessor(
            "hierarchical_modeling", self.tokenizer, self.config
        )

    def preprocess(self, document: Document) -> Dict:
        pass


    def predict(self, document: Document):
        
        self._doc_field_checker(document)

        document_prediction = []
        for page in document.pages:

            model_inputs = self.preprocess(document)
            model_outputs = self.model.predict(model_inputs)
            document_prediction.append(self.postprocess(model_outputs))
        
        return document_prediction