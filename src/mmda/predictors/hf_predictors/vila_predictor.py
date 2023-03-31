# This file rewrites the PDFPredictor classes in
# https://github.com/allenai/VILA/blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/predictors.py#L1
# to reduce the dependency on the VILA package.

from typing import List, Union, Dict, Any, Tuple
from abc import abstractmethod
from dataclasses import dataclass
import inspect
import itertools

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from vila.models.hierarchical_model import (
    HierarchicalModelForTokenClassification,
    HierarchicalModelConfig,
)
from vila.dataset.preprocessors import instantiate_dataset_preprocessor

from mmda.types.metadata import Metadata
from mmda.types.names import PagesField, RowsField, TokensField
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.predictors.hf_predictors.utils import (
    convert_document_page_to_pdf_dict,
    convert_sequence_tagging_to_spans,
    normalize_bbox,
)
from mmda.predictors.hf_predictors.base_hf_predictor import BaseHFPredictor

# Two constants for the constraining the size of the page for
# inputs to the model.
# TODO: Move this to somewhere else.
MAX_PAGE_WIDTH = 1000
MAX_PAGE_HEIGHT = 1000


def columns_used_in_model_inputs(model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    return signature_columns


@dataclass
class VILAPreprocessorConfig:
    agg_level: str = "row"
    label_all_tokens: bool = False
    group_bbox_agg: str = "first"
    added_special_sepration_token: str = "[SEP]"

    # This is introduced to support the updates in the 
    # vila 0.4.0 which fixes the typo. 
    @property
    def added_special_separation_token(self):
        return self.added_special_sepration_token


class BaseVILAPredictor(BaseHFPredictor):

    REQUIRED_BACKENDS = ["transformers", "torch", "vila"]
    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField]

    def __init__(
        self, model: Any, config: Any, tokenizer: Any, preprocessor, device=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.preprocessor = preprocessor

        if device is None:
            self.device = model.device
        else:
            self.device = device
            model.to(self.device)

        self.model.eval()
        self._used_cols = columns_used_in_model_inputs(self.model)
        # Sometimes the input data might contain certain columns that are
        # not used in the model inputs. For example, for a BERT model,
        # it won't use the `bbox` column.

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        preprocessor=None,
        device: str = None,
        **preprocessor_config
    ):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if preprocessor is None:
            preprocessor = cls.initialize_preprocessor(
                tokenizer, VILAPreprocessorConfig(**preprocessor_config)
            )

        return cls(model, config, tokenizer, preprocessor, device)

    @staticmethod
    @abstractmethod
    def initialize_preprocessor(tokenizer, config):
        # preprocessors defines how to create the actual model inputs
        # based on the raw pdf data (characterized by pdf_dict).
        # For example, in i-vila models, we can inject a special token
        # in the model inputs. This requires additional preprocessing
        # of the pdf_dicts, and it is handled by preprocessors in the
        # vila module.
        pass

    def preprocess(
        self, pdf_dict: Dict[str, List[Any]], page_width: int, page_height: int
    ) -> Dict[str, List[Any]]:
        _labels = pdf_dict.get("labels")
        pdf_dict["labels"] = [0] * len(pdf_dict["words"])
        # because the preprocess_sample requires the labels to be
        # a numeric value, so we temporarily set the labels to 0.
        # (it will set some labels to -100)
        # and we will change them back to the original labels later.

        model_inputs = self.preprocessor.preprocess_sample(pdf_dict)
        model_inputs["bbox"] = [
            [
                normalize_bbox(
                    bbox,
                    page_width,
                    page_height,
                    target_width=MAX_PAGE_WIDTH,
                    target_height=MAX_PAGE_HEIGHT,
                )
                for bbox in batch
            ]
            for batch in model_inputs["bbox"]
        ]
        pdf_dict["labels"] = _labels
        return model_inputs

    @abstractmethod
    def get_true_token_level_category_prediction(
        self, pdf_dict, model_inputs, model_predictions
    ) -> List[Union[str, int]]:
        # Typically BERT-based models will generate categories for each
        # word-piece encoded tokens (and also for included special tokens
        # like [SEP] and [CLS]). Therefore, we need to clean the predictions
        # to get the category predictions for the tokens that are actually
        # appeared inside the document.
        # The implementation of this method is specific to each model.
        pass

    def postprocess(
        self, document, pdf_dict, model_inputs, model_predictions
    ) -> List[SpanGroup]:

        true_token_prediction = self.get_true_token_level_category_prediction(
            pdf_dict, model_inputs, model_predictions
        )
        token_prediction_spans = convert_sequence_tagging_to_spans(
            true_token_prediction
        )

        prediction_spans = []
        for (token_start, token_end, label) in token_prediction_spans:
            cur_spans = document.tokens[token_start:token_end]

            start = min([ele.start for ele in cur_spans])
            end = max([ele.end for ele in cur_spans])
            sg = SpanGroup(spans=[Span(start, end)], metadata=Metadata(type=label))
            prediction_spans.append(sg)

        return prediction_spans

    def predict(self, document: Document) -> List[Annotation]:

        page_prediction_results = []
        for page_id, page in enumerate(tqdm(document.pages)):

            if page.tokens:
                page_width, page_height = document.images[page_id].size

                pdf_dict = convert_document_page_to_pdf_dict(
                    page, page_width=page_width, page_height=page_height
                )
                # VILA models trained based on absolute page width rather than the
                # size (1000, 1000) in vanilla LayoutLM models

                model_inputs = self.preprocess(pdf_dict, page_width, page_height)
                model_outputs = self.model(**self.model_input_collator(model_inputs))
                model_predictions = self.get_category_prediction(model_outputs)
                page_prediction_results.extend(
                    self.postprocess(page, pdf_dict, model_inputs, model_predictions)
                )

        return page_prediction_results

    ############################################
    ###### Some other auxiliary functions ######
    ############################################

    def get_category_prediction(self, model_outputs):
        predictions = model_outputs.logits.argmax(dim=-1).cpu().numpy()
        return predictions

    def model_input_collator(self, sample):
        return {
            key: torch.tensor(val, dtype=torch.int64, device=self.device)
            for key, val in sample.items()
            if key in self._used_cols
        }


class SimpleVILAPredictor(BaseVILAPredictor):

    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField]

    @staticmethod
    def initialize_preprocessor(tokenizer, config: VILAPreprocessorConfig):
        return instantiate_dataset_preprocessor("base", tokenizer, config)

    def get_true_token_level_category_prediction(
        self, pdf_dict, model_inputs, model_predictions
    ):
        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        true_predictions = list(itertools.chain.from_iterable(true_predictions))
        preds = [ele[0] for ele in true_predictions]
        # right here, the true_prediction has one-to-one correspondence with
        # the words in the input document.

        return preds


class IVILAPredictor(SimpleVILAPredictor):
    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField, RowsField]  # , Blocks]
    # TODO: Right now we only use the rows, but we should also use the blocks
    # in the future.

    @staticmethod
    def initialize_preprocessor(tokenizer, config):
        return instantiate_dataset_preprocessor("layout_indicator", tokenizer, config)


class HVILAPredictor(BaseVILAPredictor):
    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField, RowsField]  # , Blocks]
    # TODO: Right now we only use the rows, but we should also use the blocks
    # in the future.

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        preprocessor=None,
        device: str = None,
        **preprocessor_config
    ):
        config = HierarchicalModelConfig.from_pretrained(model_name_or_path)
        model = HierarchicalModelForTokenClassification.from_pretrained(
            model_name_or_path, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        if preprocessor is None:
            preprocessor = cls.initialize_preprocessor(
                tokenizer, VILAPreprocessorConfig(**preprocessor_config)
            )

        return cls(model, config, tokenizer, preprocessor, device)

    @staticmethod
    def initialize_preprocessor(tokenizer, config):
        return instantiate_dataset_preprocessor(
            "hierarchical_modeling", tokenizer, config
        )

    @staticmethod
    def flatten_line_level_prediction(batched_line_pred, batched_line_word_count):
        final_flattend_pred = []
        for line_pred, line_word_count in zip(
            batched_line_pred, batched_line_word_count
        ):
            assert len(line_pred) == len(line_word_count)
            for (pred, label), (line_id, count) in zip(line_pred, line_word_count):
                final_flattend_pred.append([[pred, label, line_id]] * count)

        return list(itertools.chain.from_iterable(final_flattend_pred))

    def get_true_token_level_category_prediction(
        self, pdf_dict, model_inputs, model_predictions
    ):
        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        flatten_predictions = self.flatten_line_level_prediction(
            true_predictions, model_inputs["group_word_count"]
        )

        preds = [ele[0] for ele in flatten_predictions]

        return preds
