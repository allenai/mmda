from typing import List, Union, Dict, Any, Tuple, Optional, Sequence
from abc import abstractmethod

from tqdm import tqdm
from vila.predictors import (
    SimplePDFPredictor,
    LayoutIndicatorPDFPredictor,
    HierarchicalPDFPredictor,
)

from collections import defaultdict
from smashed.interfaces.simple import (
    TokenizerMapper,
    UnpackingMapper,
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    ChangeFieldsMapper,
    Python2TorchMapper,
)
import numpy as np
import torch
import transformers

from mmda.types.names import *
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.predictors.hf_predictors.utils import (
    convert_document_page_to_pdf_dict,
    convert_sequence_tagging_to_spans,
)
from mmda.predictors.hf_predictors.base_hf_predictor import BaseHFPredictor


class BaseSinglePageTokenClassificationPredictor(BaseHFPredictor):

    REQUIRED_BACKENDS = ["transformers", "torch", "vila"]
    REQUIRED_DOCUMENT_FIELDS = [Pages, Tokens]
    DEFAULT_SUBPAGE_PER_RUN = 2 #TODO: Might remove this in the future for longformer-like models

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

                assert len(model_predictions) == len(page.tokens), f"Model predictions and tokens are not the same length ({len(model_predictions)} != {len(page.tokens)}) for page {page_id}"

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



class SmashedTokenClassificationPredictor(BaseHFPredictor):
    REQUIRED_BACKENDS = ["transformers", "torch", "smashed"]
    REQUIRED_DOCUMENT_FIELDS = ["tokens", "pages"]

    def __init__(
            self,
            model: Any,
            config: Any,
            tokenizer: Any,
            input_field: Optional[str] = 'tokens',
            batch_size: Optional[int] = 2,
            device: Optional[str] = 'cpu'
    ):
        super().__init__(model=model, config=config, tokenizer=tokenizer)

        self.input_field = input_field
        self.batch_size = batch_size
        self.device = device

        # handles tokenization, sliding window, truncation, subword to input word mapping, etc.
        self.tokenizer_mapper = TokenizerMapper(
            input_field=input_field,
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
                # 'input_ids': tokenizer.pad_token_id,      # default
                # 'attention_mask': 0,                      # default
                'word_ids': -1,
            }
        )
        # this casts python Dict[List] into tensors.  if using GPU, would do `device='gpu'`
        self.python_to_torch_mapper = Python2TorchMapper(
            device=device
        )

    @classmethod
    def from_pretrained(
            cls, model_name_or_path: str,
            input_field: Optional[str] = 'tokens',
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
                        input_field=input_field, batch_size=batch_size, device=device)
        return predictor

    def preprocess(self, document: Document) -> Sequence[Dict]:
        """Processes document into whatever makes sense for the model"""
        if self.input_field == 'tokens':
            dataset = [
                {
                    self.input_field: [token.text for token in page.tokens],
                    'page_id': page.id if page.id else i
                }
                for i, page in enumerate(document.pages)
            ]
        else:
            raise NotImplementedError(f'Currently only supports mmda.Document.tokens as input')

        dataset = self.batch_size_mapper.map(
            dataset=self.unpacking_mapper.map(
                dataset=self.tokenizer_mapper.map(
                    dataset=dataset
                )
            )
        )

        return dataset

    # TODO: kyle - unneeded, get rid of from HFPredictor
    def postprocess(self, model_outputs: defaultdict) -> List[Annotation]:
        pass

    def predict(self, document: Document) -> List[Annotation]:
        self._doc_field_checker(document)
        dataset: List[Dict] = self.preprocess(document=document)

        # every (page, word) has a prediction
        page_id_to_word_id_to_pred = defaultdict(dict)
        for i, batch in enumerate(dataset):
            preds = self._predict_batch(batch=batch)
            for pred in preds:
                page_id_to_word_id_to_pred[pred['page_id']][pred['word_id']] = (pred['label'],
                                                                                pred['score'])
        # TODO: here - make into SpanGroups
        for page in dataset:
            for word_id in range(len(page['page_tokens'])):
                page_id_to_word_id_to_pred[page['page_id']].get(word_id, None)

        import pdb; pdb.set_trace()

        annotations = self.postprocess(model_outputs=page_id_to_word_id_to_pred)
        return annotations


    def _predict_batch(self, batch: Dict) -> List[Dict]:
            #
            #   preprocessing!!  (padding & tensorification)
            #
            pytorch_batch = self.python_to_torch_mapper.transform(
                data=self.list_collator_mapper.transform(
                    data={
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask']
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
                for token_scoress, yns in zip(scores_tensor.tolist(), batch['attention_mask'])
            ]
            #
            #   postprocessing (map back to original inputs)!!
            #
            preds = []
            for j, (page_id, word_ids, token_scoress) in enumerate(zip(
                    batch['page_id'],
                    batch['word_ids'],
                    token_scoresss)
            ):
                for word_id, token_scores, is_valid_pred in zip(
                        word_ids,
                        token_scoress,
                        self._token_pooling_mask(word_ids=word_ids)
                ):
                    if word_id is None or is_valid_pred is False:
                        continue
                    else:
                        label_id = np.argmax(token_scores)
                        label_str = self.config.id2label[label_id]
                        preds.append({
                            'page_id': page_id,
                            'word_id': word_id,
                            'label': label_str,
                            'score': token_scores[label_id]
                        })
            return preds

    def _token_pooling_mask(
            self,
            token_ids: Optional[List[int]] = None,
            word_ids: Optional[List[int]] = None,
            token_scores: Optional[List[Tuple[float, float]]] = None,
            mode: str = 'first'
    ) -> List[bool]:
        """
            words are split into multiple tokens, each of which has a prediction.
            there are multiple strategies here:
            1) take only the first token prediction for whole word
            2) take the highest scoring token prediction for whole word
            3) ...
        """
        if mode == 'first':
            mask = [True]
            prev_word_id = word_ids[0]
            for current_word_id in word_ids[1:]:
                if current_word_id == prev_word_id:
                    mask.append(False)
                else:
                    mask.append(True)
                prev_word_id = current_word_id
            assert len(mask) == len(word_ids)
        else:
            raise NotImplementedError(f"mode {mode} not implemented yet")

        # if no word ID (e.g. [cls], [sep]), always mask
        mask = [tf if word_id is not None else False for tf, word_id in zip(mask, word_ids)]
        return mask


