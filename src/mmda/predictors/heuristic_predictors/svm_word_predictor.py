"""


"""

import os
import re
import tarfile
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from joblib import load
from scipy.sparse import hstack

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.predictors.heuristic_predictors.whitespace_predictor import (
    WhitespacePredictor,
)
from mmda.types import Document, Metadata, SpanGroup


class IsWordResult:
    def __init__(self, original: str, new: str, is_edit: bool) -> None:
        self.original = original
        self.new = new
        self.is_edit = is_edit


class SVMClassifier:
    def __init__(self, ohe_encoder, scaler, estimator, unigram_probs):
        self.ohe_encoder = ohe_encoder
        self.scaler = scaler
        self.estimator = estimator
        self.unigram_probs = unigram_probs
        self.default_prob = unigram_probs["<unk>"]
        self.sparse_columns = [
            "shape",
            "s_bg1",
            "s_bg2",
            "s_bg3",
            "p_bg1",
            "p_bg2",
            "p_bg3",
            "p_lower",
            "s_lower",
        ]
        self.dense_columns = [
            "p_upper",
            "s_upper",
            "p_number",
            "s_number",
            "p_isalpha",
            "s_isalpha",
            "p_len",
            "s_len",
            "multi_hyphen",
            "uni_prob",
        ]

    @classmethod
    def from_path(cls, tar_path: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tmp_dir)
                classifier = SVMClassifier.from_paths(
                    ohe_encoder_path=os.path.join(
                        tmp_dir, "svm_word_predictor/ohencoder.joblib"
                    ),
                    scaler_path=os.path.join(
                        tmp_dir, "svm_word_predictor/scaler.joblib"
                    ),
                    estimator_path=os.path.join(
                        tmp_dir, "svm_word_predictor/hyphen_clf.joblib"
                    ),
                    unigram_probs_path=os.path.join(
                        tmp_dir, "svm_word_predictor/unigram_probs.pkl"
                    ),
                )
                return classifier

    @classmethod
    def from_paths(
        cls,
        ohe_encoder_path: str,
        scaler_path: str,
        estimator_path: str,
        unigram_probs_path: str,
    ):
        ohe_encoder = load(ohe_encoder_path)
        scaler = load(scaler_path)
        estimator = load(estimator_path)
        unigram_probs = load(unigram_probs_path)
        classifier = SVMClassifier(
            ohe_encoder=ohe_encoder,
            scaler=scaler,
            estimator=estimator,
            unigram_probs=unigram_probs,
        )
        return classifier

    def batch_predict(self, words: List[str], threshold: float) -> List[IsWordResult]:
        all_features, word_id_to_feature_ids = self._get_features(words)
        all_scores = self.estimator.decision_function(all_features)
        results = []
        for word_id, feature_ids in word_id_to_feature_ids.items():
            word = words[word_id]
            word_segments = word.split("-")
            score_per_hyphen_in_word = all_scores[feature_ids]
            new_word = word_segments[0]
            for word_segment, hyphen_score in zip(
                word_segments[1:], score_per_hyphen_in_word
            ):
                if hyphen_score > threshold:
                    new_word += "-"
                else:
                    new_word += ""
                new_word += word_segment
            results.append(
                IsWordResult(original=word, new=new_word, is_edit=word != new_word)
            )
        return results

    def _get_dense_features(self, part: str, name_prefix: str):
        upper = int(part[0].isupper())
        number = int(part.isnumeric())
        alpha = int(part.isalpha())
        lower = part.lower()
        plen = len(part)
        return {
            f"{name_prefix}_upper": upper,
            f"{name_prefix}_number": number,
            f"{name_prefix}_isalpha": alpha,
            f"{name_prefix}_len": plen,
        }

    def _get_features(self, words: List[str]):
        sparse_all, dense_all = [], []
        idx, word_id_to_feature_ids = 0, dict()
        for widx, word in enumerate(words):
            split = word.split("-")
            for i, s in enumerate(split[:-1]):
                sparse_features, dense_features = dict(), dict()
                prefix = "-".join(split[: i + 1])
                suffix = "-".join(split[i + 1 :])
                if widx not in word_id_to_feature_ids:
                    word_id_to_feature_ids[widx] = []
                word_id_to_feature_ids[widx].append(idx)
                idx += 1
                dense_features.update(self._get_dense_features(prefix, "p"))
                dense_features.update(self._get_dense_features(suffix, "s"))
                orig_uni_prob = self.unigram_probs.get(word, self.default_prob)
                presuf_uni_prob = self.unigram_probs.get(
                    f"{prefix}{suffix}", self.default_prob
                )
                dense_features["uni_prob"] = orig_uni_prob - presuf_uni_prob
                dense_features["multi_hyphen"] = int(word.count("-") > 1)
                sparse_features["shape"] = re.sub("\w", "x", word)
                sparse_features["s_lower"] = suffix.lower()
                sparse_features["s_bg1"] = suffix[:2]
                sparse_features["s_bg2"] = suffix[1:3] if len(suffix) > 2 else ""
                sparse_features["s_bg3"] = suffix[2:4] if len(suffix) > 3 else ""
                sparse_features["p_lower"] = prefix.lower()
                sparse_features["p_bg1"] = prefix[-2:][::-1] if len(prefix) > 1 else ""
                sparse_features["p_bg2"] = (
                    prefix[-3:-1][::-1] if len(prefix) > 2 else ""
                )
                sparse_features["p_bg3"] = (
                    prefix[-4:-2][::-1] if len(prefix) > 3 else ""
                )
                sparse_all.append([sparse_features[k] for k in self.sparse_columns])
                dense_all.append([dense_features[k] for k in self.dense_columns])
        dense_transformed = self.scaler.transform(dense_all)
        sparse_transformed = self.ohe_encoder.transform(sparse_all)

        return hstack([sparse_transformed, dense_transformed]), word_id_to_feature_ids


class SVMWordPredictor(BasePredictor):
    def __init__(self, classifier: SVMClassifier, threshold: float = -1.5):
        self.classifier = classifier
        self.whitespace_predictor = WhitespacePredictor()
        self.threshold = threshold

    @classmethod
    def from_path(cls, tar_path: str):
        classifier = SVMClassifier.from_path(tar_path=tar_path)
        predictor = SVMWordPredictor(classifier=classifier)
        return predictor

    def predict(self, document: Document) -> List[SpanGroup]:
        # validate input
        self._validate_tokenization(document=document)

        # initialize output data using whitespace tokenization
        (
            token_id_to_word_id,
            word_id_to_token_ids,
            word_id_to_text,
        ) = self._predict_with_whitespace(document=document)

        # get hyphen word candidates
        hyphen_word_candidates = self._find_hyphen_word_candidates(
            tokens=document.tokens,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_token_ids=word_id_to_token_ids,
            word_id_to_text=word_id_to_text,
        )

        # classify hyphen words
        candidate_texts = [
            word_id_to_text[prefix_word_id] + word_id_to_text[suffix_word_id]
            for prefix_word_id, suffix_word_id in hyphen_word_candidates
        ]
        results = self.classifier.batch_predict(
            words=candidate_texts, threshold=self.threshold
        )

        # update output data based on hyphen word candidates
        # first, we concatenate words based on prefix + suffix. this includes hyphen.
        # second, we modify the text value (e.g. remove hyphens) if classifier says.
        for (prefix_word_id, suffix_word_id), result in zip(
            hyphen_word_candidates, results
        ):
            impacted_token_ids = (
                word_id_to_token_ids[prefix_word_id]
                + word_id_to_token_ids[suffix_word_id]
            )
            word_id_to_token_ids[prefix_word_id] = impacted_token_ids
            word_id_to_token_ids.pop(suffix_word_id)
            word_id_to_text[prefix_word_id] += word_id_to_text[suffix_word_id]
            word_id_to_text.pop(suffix_word_id)
            if result.is_edit is True:
                word_id_to_text[prefix_word_id] = result.new
        token_id_to_word_id = {
            token_id: word_id
            for word_id, token_ids in word_id_to_token_ids.items()
            for token_id in token_ids
        }

        # make into spangroups
        words = self._create_words(
            document=document,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_text=word_id_to_text,
        )
        return words

    def _recursively_remove_trailing_hyphens(self, word: str) -> str:
        if word.endswith("-"):
            return self._recursively_remove_trailing_hyphens(word=word[:-1])
        else:
            return word

    def _validate_tokenization(self, document: Document):
        """This Word Predictor relies on a specific type of Tokenization
        in which hyphens ('-') must be their own token. This verifies.

        Additionally, doesnt work unless there's an `.id` field on each token.
        See `_cluster_tokens_by_whitespace()` for more info.
        """
        for token in document.tokens:
            if "-" in token.text and token.text != "-":
                raise ValueError(
                    f"Document contains Token with hyphen, but not as its own token."
                )
            if token.id is None:
                raise ValueError(
                    f"Document contains Token without an `.id` field, which is necessary for this word Predictor's whitespace clustering operation."
                )

    def _cluster_tokens_by_whitespace(self, document: Document) -> List[List[int]]:
        """
        `whitespace_tokenization` is necessary because lack of whitespace is an indicator that
        adjacent tokens belong in a word together.
        """
        _ws_tokens: List[SpanGroup] = self.whitespace_predictor.predict(
            document=document
        )
        document.annotate(_ws_tokens=_ws_tokens)

        # token -> ws_tokens
        token_id_to_ws_token_id = {}
        for token in document.tokens:
            token_id_to_ws_token_id[token.id] = token._ws_tokens[0].id

        # ws_token -> tokens
        ws_token_id_to_tokens = defaultdict(list)
        for token_id, ws_token_id in token_id_to_ws_token_id.items():
            ws_token_id_to_tokens[ws_token_id].append(token_id)

        # token -> all cluster tokens (inclusive)
        # token_id_to_token_ids = {}
        # for token_id, ws_token_id in token_id_to_ws_token_id.items():
        #     candidate_token_ids = [i for i in ws_token_id_to_tokens[ws_token_id]]
        #     token_id_to_token_ids[token_id] = candidate_token_ids

        # cluster tokens by whitespace
        clusters = [
            sorted(ws_token_id_to_tokens[ws_token_id])
            for ws_token_id in range(len(ws_token_id_to_tokens))
        ]

        # cleanup
        document.remove("_ws_tokens")
        return clusters

    def _predict_with_whitespace(self, document: Document):
        """Predicts word boundaries using whitespace tokenization."""
        # precompute whitespace tokenization
        whitespace_clusters = self._cluster_tokens_by_whitespace(document=document)
        # assign word ids
        token_id_to_word_id = {}
        word_id_to_token_ids = defaultdict(list)
        for word_id, token_ids_in_cluster in enumerate(whitespace_clusters):
            for token_id in token_ids_in_cluster:
                token_id_to_word_id[token_id] = word_id
                word_id_to_token_ids[word_id].append(token_id)
        # get word strings
        word_id_to_text = {}
        for word_id, token_ids in word_id_to_token_ids.items():
            word_id_to_text[word_id] = "".join(
                document.tokens[token_id].text for token_id in token_ids
            )
        return token_id_to_word_id, word_id_to_token_ids, word_id_to_text

    def _find_hyphen_word_candidates(
        self,
        tokens,
        token_id_to_word_id,
        word_id_to_token_ids,
        word_id_to_text,
    ) -> Tuple[int, int]:
        """Finds the IDs of hyphenated words (in prefix + suffix format)."""
        # get all hyphen tokens
        # TODO: can refine this further by restricting to only tokens at end of `rows`
        hyphen_token_ids = []
        for token_id, token in enumerate(tokens):
            if token.text == "-":
                hyphen_token_ids.append(token_id)
        # get words that contain hyphen token, but only at the end (i.e. broken word)
        # these form the `prefix` of a potential hyphenated word
        prefix_word_ids = set()
        for hyphen_token_id in hyphen_token_ids:
            prefix_word_id = token_id_to_word_id[hyphen_token_id]
            prefix_word_text = word_id_to_text[prefix_word_id]
            if prefix_word_text.endswith("-"):
                prefix_word_ids.add(prefix_word_id)
        # get words right after the prefix (assumed words in reading order)
        # these form the `suffix` of a potential hyphenated word
        # together, a `prefix` and `suffix` form a candidate pair
        word_id_pairs = []
        for prefix_word_id in prefix_word_ids:
            suffix_word_id = prefix_word_id + 1
            suffix_word_text = word_id_to_text[suffix_word_id]
            word_id_pairs.append((prefix_word_id, suffix_word_id))
        return sorted(word_id_pairs)

    def _create_words(
        self, document: Document, token_id_to_word_id, word_id_to_text
    ) -> List[SpanGroup]:
        words = []
        tokens_in_word = [document.tokens[0]]
        current_word_id = 0
        for token_id in range(1, len(token_id_to_word_id)):
            token = document.tokens[token_id]
            word_id = token_id_to_word_id[token_id]
            if word_id == current_word_id:
                tokens_in_word.append(token)
            else:
                word = SpanGroup(
                    spans=[span for token in tokens_in_word for span in token.spans],
                    id=current_word_id,
                    metadata=Metadata(text=word_id_to_text[current_word_id]),
                )
                words.append(word)
                tokens_in_word = [token]
                current_word_id = word_id
        # last bit
        word = SpanGroup(
            spans=[span for token in tokens_in_word for span in token.spans],
            id=current_word_id,
            metadata=Metadata(text=word_id_to_text[current_word_id]),
        )
        words.append(word)
        # cleanup the word IDs so no skipped integers
        for i, word in enumerate(words):
            word.id = i
        return words
