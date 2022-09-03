from typing import List, Union, Dict, Any, Tuple, Optional, Sequence
from abc import abstractmethod

from tqdm import tqdm
from vila.predictors import (
    SimplePDFPredictor,
    LayoutIndicatorPDFPredictor,
    HierarchicalPDFPredictor,
)

import itertools
from collections import defaultdict
from smashed.interfaces.simple import (
    TokenizerMapper,
    UnpackingMapper,
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    Python2TorchMapper,
)
import numpy as np
import torch
import transformers

from mmda.types.names import *
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.types.metadata import Metadata
from mmda.predictors.hf_predictors.utils import (
    convert_document_page_to_pdf_dict,
    convert_sequence_tagging_to_spans,
)
from mmda.predictors.hf_predictors.base_hf_predictor import BaseHFPredictor


class BaseSinglePageTokenClassificationPredictor(BaseHFPredictor):
    REQUIRED_BACKENDS = ["transformers", "torch", "vila"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Tokens]
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
            prediction_spans.append(SpanGroup(spans=[Span(start, end)], type=label))

        return prediction_spans


class SinglePageTokenClassificationPredictor(
    BaseSinglePageTokenClassificationPredictor
):
    VILA_MODEL_CLASS = SimplePDFPredictor


class IVILATokenClassificationPredictor(BaseSinglePageTokenClassificationPredictor):
    VILA_MODEL_CLASS = LayoutIndicatorPDFPredictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List:
        base_reqs = [Pages, Tokens]
        if self.predictor.preprocessor.config.agg_level == "row":
            base_reqs.append(Rows)
        elif self.predictor.preprocessor.config.agg_level == "block":
            base_reqs.append(Blocks)
        return base_reqs


class HVILATokenClassificationPredictor(BaseSinglePageTokenClassificationPredictor):
    VILA_MODEL_CLASS = HierarchicalPDFPredictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List:
        base_reqs = [Pages, Tokens]
        if self.predictor.preprocessor.config.agg_level == "row":
            base_reqs.append(Rows)
        elif self.predictor.preprocessor.config.agg_level == "block":
            base_reqs.append(Blocks)
        return base_reqs


class SpanGroupClassificationBatch:
    def __init__(
            self,
            input_ids: List[List[int]],
            attention_mask: List[List[int]],
            span_group_ids: List[List[Optional[int]]],
            page_id: List[int]
    ):
        assert len(input_ids) == len(attention_mask) == len(span_group_ids) == len(page_id), \
            f"Inputs to batch arent same length"
        self.batch_size = len(input_ids)
        assert [len(example) for example in input_ids] == \
               [len(example) for example in attention_mask] == \
               [len(example) for example in span_group_ids], f"Examples in batch arent same length"
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.span_group_ids = span_group_ids
        self.page_id = page_id


class SpanGroupClassificationPrediction:
    def __init__(self, page_id: int, span_group_id: int, label: str, score: float):
        self.page_id = page_id
        self.span_group_id = span_group_id
        self.label = label
        self.score = score


class SpanGroupClassificationPredictor(BaseHFPredictor):
    REQUIRED_BACKENDS = ["transformers", "torch", "smashed"]
    REQUIRED_DOCUMENT_FIELDS = ["pages"]

    _SPAN_GROUP = 'inputs'
    _PAGE_ID = 'page_id'

    def __init__(
            self,
            model: Any,
            config: Any,
            tokenizer: Any,
            span_group_name: str,
            batch_size: Optional[int] = 2,
            device: Optional[str] = 'cpu'
    ):
        super().__init__(model=model, config=config, tokenizer=tokenizer)

        self.span_group_name = span_group_name
        self.batch_size = batch_size
        self.device = device

        # handles tokenization, sliding window, truncation, subword to input word mapping, etc.
        self.tokenizer_mapper = TokenizerMapper(
            input_field=self._SPAN_GROUP,
            tokenizer=tokenizer,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
            return_overflowing_tokens=True,
            return_word_ids=True
        )
        # since input data is automatically chunked into segments (e.g. 512 length),
        # each example <dict> actually becomes many input sequences.
        # this mapper unpacks all of this into one <dict> per input sequence.
        # we set `repeat` because we want other fields (e.g., `page_id`) to repeat across sequnces
        self.unpacking_mapper = UnpackingMapper(
            fields_to_unpack=['input_ids', 'attention_mask', 'word_ids'],
            ignored_behavior='repeat'
        )
        # at the end of this, each <dict> contains <lists> of length `batch_size`
        # where each element is variable length within the `max_length` limit.
        # `keep_last` controls whether we want partial batches, which we always do
        # for token classification (i.e. we dont want to miss anything!)
        self.batch_size_mapper = FixedBatchSizeMapper(
            batch_size=batch_size,
            keep_last=True
        )
        # this performs padding so all sequences in a batch are of same length
        self.list_collator_mapper = FromTokenizerListCollatorMapper(
            tokenizer=tokenizer,
            pad_to_length=None,  # keeping this `None` is best because dynamic padding
            fields_pad_ids={
                'word_ids': -1,  # avoid word_ids which are <int> >= 0. can be any number tbh...
            }
        )
        # this casts python Dict[List] into tensors.  if using GPU, would do `device='gpu'`
        self.python_to_torch_mapper = Python2TorchMapper(
            device=device
        )
        # combining everything
        self.preprocess_mapper = self.tokenizer_mapper >> \
                                 self.unpacking_mapper >> \
                                 self.batch_size_mapper

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path: str,
            span_group_name: str,
            batch_size: Optional[int] = 2,
            device: Optional[str] = 'cpu',
            *args,
            **kwargs
    ):
        """If `model_name_or_path` is a path, should be a directory
        containing `vocab.txt`, `config.json`, and `pytorch_model.bin`

        NOTE: slightly annoying, but if loading in this way, the `_name_or_path`
        in `model.config` != `config`.
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )
        predictor = cls(model, config, tokenizer,
                        span_group_name=span_group_name, batch_size=batch_size, device=device)
        return predictor

    def preprocess(self, document: Document) -> Sequence[SpanGroupClassificationBatch]:
        """Processes document into whatever makes sense for the Huggingface model"""
        # (1) get it into a dictionary format that Smashed expects
        dataset = [
            {
                self._SPAN_GROUP: [sg.text for sg in getattr(page, self.span_group_name)],
                self._PAGE_ID: i
            }
            for i, page in enumerate(document.pages)
        ]
        # (2) get it
        return [
            # slightly annoying, but the names `input_ids`, `attention_mask` and `word_ids` are
            # reserved and produced after tokenization, which is why hard-coded here.
            SpanGroupClassificationBatch(
                input_ids=batch_dict['input_ids'],
                attention_mask=batch_dict['attention_mask'],
                span_group_ids=batch_dict['word_ids'],
                page_id=batch_dict[self._PAGE_ID]
            ) for batch_dict in self.preprocess_mapper.map(dataset=dataset)
        ]

    def postprocess(
            self,
            doc: Document,
            preds: List[SpanGroupClassificationPrediction]
    ) -> List[Annotation]:
        """This function handles a bunch of nonsense that happens with Huggingface models &
        how we processed the data.  Namely:

        Because Huggingface might drop tokens during the course of tokenization
        we need to organize our predictions into a Lookup <dict> and cross-reference
        with the original input SpanGroups to make sure they all got classified.
        """
        # (1) organize predictions into a Lookup at the (Page, SpanGroup) level.
        page_id_to_span_group_id_to_pred = defaultdict(dict)
        for pred in preds:
            page_id_to_span_group_id_to_pred[pred.page_id][pred.span_group_id] = pred

        # (2) iterate through original data to check against that Lookup
        annotations: List[Annotation] = []
        for i, page in enumerate(doc.pages):
            for j, span_group in enumerate(getattr(page, self.span_group_name)):
                pred = page_id_to_span_group_id_to_pred[i].get(j, None)
                if pred is not None:
                    new_metadata = Metadata.from_json(span_group.metadata.to_json())
                    new_metadata.type = pred.label
                    new_metadata.score = pred.score
                    new_span_group = SpanGroup(
                        spans=span_group.spans,
                        box_group=span_group.box_group,
                        # TODO: double-check whether this deepcopy is needed...
                        metadata=new_metadata
                    )
                    annotations.append(new_span_group)
        return annotations

    def predict(self, document: Document) -> List[Annotation]:
        self._doc_field_checker(document)

        # (1) Make batches
        batches: List[SpanGroupClassificationBatch] = self.preprocess(document=document)

        # (2) Predict each batch.
        preds: List[SpanGroupClassificationPrediction] = []
        for batch in batches:
            for pred in self._predict_batch(batch=batch):
                preds.append(pred)

        # (3) Postprocess into proper Annotations
        annotations = self.postprocess(doc=document, preds=preds)
        return annotations

    def _predict_batch(
            self,
            batch: SpanGroupClassificationBatch
    ) -> List[SpanGroupClassificationPrediction]:
        #
        #   preprocessing!!  (padding & tensorification)
        #
        pytorch_batch = self.python_to_torch_mapper.transform(
            data=self.list_collator_mapper.transform(
                data={
                    'input_ids': batch.input_ids,
                    'attention_mask': batch.attention_mask
                }
            )
        )
        #
        #   inference!! (preferably on gpu)
        #
        # TODO: add something here for gpu migration
        pytorch_output = self.model(**pytorch_batch)
        scores_tensor = torch.softmax(pytorch_output.logits, dim=2)
        token_scoresss = [
            [
                token_scores for token_scores, yn in zip(token_scoress, yns)
                if yn == 1
            ]
            for token_scoress, yns in zip(scores_tensor.tolist(), batch.attention_mask)
        ]
        #
        #   postprocessing (map back to original inputs)!!
        #
        preds = []
        for j, (page_id, word_ids, token_scoress) in enumerate(zip(
                batch.page_id,
                batch.span_group_ids,
                token_scoresss)
        ):
            for word_id, token_scores, is_valid_pred in zip(
                    word_ids,
                    token_scoress,
                    self._token_pooling_strategy_mask(word_ids=word_ids)
            ):
                if word_id is None or is_valid_pred is False:
                    continue
                else:
                    label_id = np.argmax(token_scores)
                    pred = SpanGroupClassificationPrediction(
                        page_id=page_id,
                        span_group_id=word_id,
                        label=self.config.id2label[label_id],
                        score=token_scores[label_id]
                    )
                    preds.append(pred)
        return preds

    def _token_pooling_strategy_mask(
            self,
            token_ids: Optional[List[int]] = None,
            word_ids: Optional[List[int]] = None,
            token_scores: Optional[List[Tuple[float, float]]] = None,
            strategy: str = 'first'
    ) -> List[bool]:
        """
        words are split into multiple tokens, each of which has a prediction.
        there are multiple strategies to decide the model prediction at a word-level:
        1) 'first': take only the first token prediction for whole word
        2) 'max': take the highest scoring token prediction for whole word
        3) ...
        """
        if strategy == 'first':
            mask = [True]
            prev_word_id = word_ids[0]
            for current_word_id in word_ids[1:]:
                if current_word_id == prev_word_id:
                    mask.append(False)
                else:
                    mask.append(True)
                prev_word_id = current_word_id
        else:
            raise NotImplementedError(f"mode {strategy} not implemented yet")

        # if no word ID (e.g. [cls], [sep]), always mask
        mask = [
            is_word if word_id is not None else False
            for is_word, word_id in zip(mask, word_ids)
        ]
        return mask




        #
        # def _convert_token_preds_to_spans(
        #     self,
        #     token_preds: List[SpanGroupClassificationPrediction]
        # ) -> List[PredictedSpan]:
        #     """For a sequence of token predictions, convert them to spans
        #     of consecutive same predictions."""
        #     spans = []
        #     for label, subsequence in itertools.groupby(token_preds, key=lambda p: p.label):
        #         subsequence = list(subsequence)
        #         print(f"{label}\t{subsequence}")
        #         span = PredictedSpan(span=Span.small_spans_to_big_span(spans=),
        #                              label=label,
        #                              score=0.00)
        #         # spans.append(span)
        #         # prev_len = prev_len + cur_len
        #     import pdb;pdb.set_trace()
        #     return spans
        #
