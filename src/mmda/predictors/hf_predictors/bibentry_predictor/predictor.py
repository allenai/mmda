import os
import re
from typing import Dict, List, Optional, Tuple

from optimum.onnxruntime import ORTModelForTokenClassification
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from unidecode import unidecode

from mmda.predictors.hf_predictors.base_hf_predictor import BasePredictor
from mmda.predictors.hf_predictors.bibentry_predictor import utils
from mmda.predictors.hf_predictors.bibentry_predictor.types import (
    BibEntryLabel,
    BibEntryPredictionWithSpan,
    BibEntryStructureSpanGroups,
    StringWithSpan
)
from mmda.types.document import Document


class BibEntryPredictor(BasePredictor):

    REQUIRED_BACKENDS = ["transformers", "torch"]
    REQUIRED_DOCUMENT_FIELDS = ["tokens", "pages", "bib_entry_boxes"]

    def __init__(self, model_name_or_path: str):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        onnx = os.path.exists(os.path.join(model_name_or_path, "model.onnx"))
        if onnx:
            self.model = ORTModelForTokenClassification.from_pretrained(model_name_or_path, file_name="model.onnx")
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if not onnx:
            # https://stackoverflow.com/a/60018731
            self.model.eval()  # for some reason the onnx version doesnt have an eval()

    def predict(self, document: Document) -> BibEntryStructureSpanGroups:
        # Recover the (approximate) raw bibentry strings from mmda document
        bib_entry_strings = utils.mk_bib_entry_strings(document)

        # Delegate to underlying model for inference
        raw_predictions = self.predict_raw(bib_entry_strings)

        # Map raw predictions back into valid annotations for passed document
        prediction = utils.map_raw_predictions_to_mmda(document.bib_entry_boxes, raw_predictions)

        return prediction

    def predict_raw(self, bib_entries: List[str]) -> List[BibEntryPredictionWithSpan]:
        if not bib_entries:
            return []

        res = []

        tokenized_inputs = self.tokenizer(bib_entries, padding=True, truncation=True, return_tensors="pt")
        # put the data on the same device of the model.
        tokenized_inputs = tokenized_inputs.to(self.model.device)

        predictions = self.model(**tokenized_inputs)

        pred_ids = predictions.logits.argmax(2).tolist()

        num_items = len(bib_entries)

        for i in range(num_items):
            # Combine token-level prediction into word-level prediction
            label_ids = BibEntryPredictor._get_word_level_prediction(tokenized_inputs.word_ids(i), pred_ids[i])
            word_ids = [id for id in tokenized_inputs.word_ids(i) if id is not None]
            num_words = word_ids[-1] + 1 if word_ids else 0
            spans = [tokenized_inputs.word_to_chars(i, word_index) for word_index in range(num_words)]

            # Extract output fields from word predictions
            res.append(BibEntryPredictor._aggregate_token_level_prediction(bib_entries[i], spans, label_ids))

        return res

    @staticmethod
    def postprocess(pred: BibEntryPredictionWithSpan) -> Dict:
        citation_number = pred.citation_number.content if pred.citation_number else None
        title = BibEntryPredictor._clean_str(pred.title.content) if pred.title else None
        doi = BibEntryPredictor._clean_doi(pred.doi.content) if pred.doi else None
        return dict(
            citation_number=citation_number,
            title=title,
            doi=doi
        )

    @staticmethod
    def _get_word_level_prediction(word_ids: List[Optional[int]], predictions: List[int]) -> List[int]:
        """
        If a word is split into 2 or more tokens, only take prediction for the first token.
        """
        res = []
        prev_word_id = None
        for word_id, pred in zip(word_ids, predictions):
            if word_id is not None and word_id != prev_word_id:
                # Tokenization process removes empty string and skips word id, so we're adding it back here
                # For example:
                # input string list: [' Anon ', '1934', ' ', 'University and Educational Intelligence', ' ', 'Nature', ' ', '133', ' ', '805–805']
                # tokenization removes empty string: ['[CLS]', 'an', '##on', '1934', 'university', 'and', 'educational', 'intelligence', 'nature', '133', '80', '##5', '–', '80', '##5', '[SEP]']
                # skipping empty string results in skipping word id: [None, 0, 0, 1, 3, 3, 3, 3, 5, 7, 9, 9, 9, 9, 9, None]
                # predictions: [0, 9, 9, 0, 8, 9, 8, 8, 9, 0, 13, 13, 13, 13, 13, 4]
                if prev_word_id is not None:
                    for i in range(word_id - (prev_word_id + 1)):
                        res.append(BibEntryLabel.MISC.value)

                res.append(pred)
            prev_word_id = word_id
        return res

    @staticmethod
    def _aggregate_token_level_prediction(input: str, spans, label_ids: List[int]) -> BibEntryPredictionWithSpan:
        citation_number = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids,
                                                                                              BibEntryLabel.CITATION_NUMBER)

        authors = BibEntryPredictor._extract_author_token(input, spans, label_ids)
        title = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids, BibEntryLabel.TITLE)

        journal = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids,
                                                                                      BibEntryLabel.JOURNAL)
        event = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids, BibEntryLabel.EVENT)
        journal_venue_or_event = journal if journal else event

        year = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids,
                                                                                   BibEntryLabel.ISSUED_YEAR)
        doi = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids, BibEntryLabel.DOI)
        url = BibEntryPredictor._extract_first_contiguous_label_group_token_level(input, spans, label_ids, BibEntryLabel.URL)

        return BibEntryPredictionWithSpan(
            citation_number=citation_number,
            authors=authors,
            title=title,
            journal_venue_or_event=journal_venue_or_event,
            year=year,
            doi=doi,
            url=url
        )

    @staticmethod
    def _extract_author_token(input: str, spans, label_ids: List[int]) -> Optional[List[StringWithSpan]]:
        res = []
        author_span = None

        for word_index, label_id in enumerate(label_ids):
            # Beginning of new author
            if label_id == BibEntryLabel.AUTHOR_START.value and not author_span:
                author_span = spans[word_index]
            # Middle of current author
            elif (
                    label_id == BibEntryLabel.AUTHOR_START.value or label_id == BibEntryLabel.AUTHOR_MIDDLE.value or label_id == BibEntryLabel.AUTHOR_END.value) and author_span:
                current_span = spans[word_index]
                author_span = author_span._replace(end=current_span.end)
            # End of current author. Close current author span and reset.
            elif (
                    label_id != BibEntryLabel.AUTHOR_START.value and label_id != BibEntryLabel.AUTHOR_MIDDLE.value and label_id != BibEntryLabel.AUTHOR_END.value) and author_span:
                res.append(StringWithSpan(
                    content=input[author_span.start:author_span.end],
                    start=author_span.start,
                    end=author_span.end,
                ))
                author_span = None

        return res if res else None

    @staticmethod
    def _extract_first_contiguous_label_group_token_level(
            input: str,
            spans,
            label_ids: List[int],
            target_label: BibEntryLabel
    ) -> Optional[StringWithSpan]:
        res = None
        existing_span = None
        for word_index, label_id in enumerate(label_ids):
            if label_id == target_label.value:
                # Middle of label span
                if existing_span:
                    current_span = spans[word_index]
                    existing_span = existing_span._replace(end=current_span.end)
                    # First label encounter
                else:
                    existing_span = spans[word_index]
            # End of label span
            elif existing_span:
                break

        if existing_span:
            res = StringWithSpan(
                content=input[existing_span.start:existing_span.end],
                start=existing_span.start,
                end=existing_span.end,
            )

        return res

    @staticmethod
    def _clean_str(s: str) -> Optional[str]:
        without_diacritics = unidecode(s.strip())
        subbed = re.sub("-\s+", "", without_diacritics)
        if subbed:
            return subbed
        else:
            return None

    @staticmethod
    def _clean_doi(doi: str) -> Optional[str]:
        lower_trimmed = doi.strip().lower()
        if lower_trimmed.startswith("10."):
            return re.sub("\s", "", lower_trimmed)
        else:
            return None
