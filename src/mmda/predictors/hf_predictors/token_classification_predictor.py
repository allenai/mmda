from typing import List, Union, Dict, Any, Tuple, Optional, Sequence
from abc import abstractmethod

from tqdm import tqdm
from vila.predictors import (
    SimplePDFPredictor,
    LayoutIndicatorPDFPredictor,
    HierarchicalPDFPredictor,
)


from mmda.types.metadata import Metadata
from mmda.types.names import BlocksField, PagesField, RowsField, TokensField
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.predictors.hf_predictors.utils import (
    convert_document_page_to_pdf_dict,
    convert_sequence_tagging_to_spans,
)
from mmda.predictors.hf_predictors.base_hf_predictor import BaseHFPredictor


class BaseSinglePageTokenClassificationPredictor(BaseHFPredictor):
    REQUIRED_BACKENDS = ["transformers", "torch", "vila"]
    REQUIRED_DOCUMENT_FIELDS = [PagesField, TokensField]
    DEFAULT_SUBPAGE_PER_RUN = 2  # TODO: Might remove this in the future for longformer-like models

    @property
    @abstractmethod
    def VILA_MODEL_CLASS(self):
        pass

    def __init__(self, predictor, subpage_per_run: Optional[int] = None):

        self.predictor = predictor

        # TODO: Make this more robust
        self.id2label = self.predictor.model.config.id2label
        self.label2id = self.predictor.model.config.label2id

        self.subpage_per_run = subpage_per_run or self.DEFAULT_SUBPAGE_PER_RUN

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path: str,
            preprocessor=None,
            device: Optional[str] = None,
            subpage_per_run: Optional[int] = None,
            **preprocessor_config
    ):
        predictor = cls.VILA_MODEL_CLASS.from_pretrained(
            model_path=model_name_or_path,
            preprocessor=preprocessor,
            device=device,
            **preprocessor_config
        )

        return cls(predictor, subpage_per_run)

    def predict(
            self, document: Document, subpage_per_run: Optional[int] = None
    ) -> List[Annotation]:

        page_prediction_results = []
        for page_id, page in enumerate(tqdm(document.pages)):

            if page.tokens:
                page_width, page_height = document.images[page_id].size

                pdf_dict = self.preprocess(
                    page, page_width=page_width, page_height=page_height
                )

                model_predictions = self.predictor.predict(
                    page_data=pdf_dict,
                    page_size=(page_width, page_height),
                    batch_size=subpage_per_run or self.subpage_per_run,
                    return_type="list",
                )

                assert len(model_predictions) == len(
                    page.tokens), f"Model predictions and tokens are not the same length ({len(model_predictions)} != {len(page.tokens)}) for page {page_id}"

                page_prediction_results.extend(
                    self.postprocess(page, model_predictions)
                )

        return page_prediction_results

    def preprocess(self, page: Document, page_width: float, page_height: float) -> Dict:

        # In the latest vila implementations (after 0.4.0), the predictor will
        # handle all other preprocessing steps given the pdf_dict input format.

        return convert_document_page_to_pdf_dict(
            page, page_width=page_width, page_height=page_height
        )

    def postprocess(self, document: Document, model_predictions) -> List[SpanGroup]:

        token_prediction_spans = convert_sequence_tagging_to_spans(model_predictions)

        prediction_spans = []
        for (token_start, token_end, label) in token_prediction_spans:
            cur_spans = document.tokens[token_start:token_end]

            start = min([ele.start for ele in cur_spans])
            end = max([ele.end for ele in cur_spans])
            sg = SpanGroup(spans=[Span(start, end)], metadata=Metadata(type=label))
            prediction_spans.append(sg)
        return prediction_spans


class SinglePageTokenClassificationPredictor(
    BaseSinglePageTokenClassificationPredictor
):
    VILA_MODEL_CLASS = SimplePDFPredictor


class IVILATokenClassificationPredictor(BaseSinglePageTokenClassificationPredictor):
    VILA_MODEL_CLASS = LayoutIndicatorPDFPredictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List:
        base_reqs = [PagesField, TokensField]
        if self.predictor.preprocessor.config.agg_level == "row":
            base_reqs.append(RowsField)
        elif self.predictor.preprocessor.config.agg_level == "block":
            base_reqs.append(BlocksField)
        return base_reqs


class HVILATokenClassificationPredictor(BaseSinglePageTokenClassificationPredictor):
    VILA_MODEL_CLASS = HierarchicalPDFPredictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List:
        base_reqs = [PagesField, TokensField]
        if self.predictor.preprocessor.config.agg_level == "row":
            base_reqs.append(RowsField)
        elif self.predictor.preprocessor.config.agg_level == "block":
            base_reqs.append(BlocksField)
        return base_reqs

