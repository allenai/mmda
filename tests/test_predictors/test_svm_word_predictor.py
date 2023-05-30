"""
Tests for SVM Word Predictor

@kylel
"""

import json
import os
import unittest
from typing import List, Optional, Set

import numpy as np

from mmda.predictors.heuristic_predictors.svm_word_predictor import SVMWordPredictor
from mmda.types import Document, Span, SpanGroup


class TestSVMWordPredictor(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.fixture_path = os.path.join(os.path.dirname(__file__), "../fixtures/")
        self.predictor = SVMWordPredictor.from_path(
            tar_path=os.path.join(
                self.fixture_path, "svm_word_predictor/svm_word_predictor.tar.gz"
            )
        )
        with open(
            os.path.join(self.fixture_path, "svm_word_predictor/pos_words.txt")
        ) as f_in:
            self.pos_words = [line.strip() for line in f_in]
        with open(
            os.path.join(self.fixture_path, "svm_word_predictor/neg_words.txt")
        ) as f_in:
            self.neg_words = [line.strip() for line in f_in]

        with open(
            os.path.join(
                self.fixture_path,
                "types/20fdafb68d0e69d193527a9a1cbe64e7e69a3798__pdfplumber_doc.json",
            )
        ) as f_in:
            doc_dict = json.load(f_in)
            self.doc = Document.from_json(doc_dict=doc_dict)

    def test_predict(self):
        words = self.predictor.predict(document=self.doc)
        # double-check number of units
        self.assertLess(len(words), len(self.doc.tokens))
        # double-check can annotate
        self.doc.annotate(words=words)
        # after annotating, double-check words against tokens
        for word in self.doc.words:
            tokens_in_word = word.tokens
            # if word is a single token, then it should be the same as the token
            if len(tokens_in_word) == 1:
                self.assertEqual(tokens_in_word[0].text, word.text)
                self.assertEqual(tokens_in_word[0].spans, word.spans)
            else:
                # otherwise, most token units should be substring of word
                overlap = np.mean([token.text in word.text for token in tokens_in_word])
                self.assertGreaterEqual(overlap, 0.5)
                # spans should match up though
                self.assertEqual(
                    [span for token in tokens_in_word for span in token.spans],
                    word.spans,
                )

    def test_validate_tokenization(self):
        doc = Document.from_json(
            doc_dict={
                "symbols": "I am the wizard-of-oz.",
                "tokens": [
                    {"id": 0, "spans": [{"start": 0, "end": 1}]},
                    {"id": 1, "spans": [{"start": 2, "end": 4}]},
                    {"id": 2, "spans": [{"start": 5, "end": 8}]},
                    {"id": 3, "spans": [{"start": 9, "end": 15}]},
                    {"id": 4, "spans": [{"start": 15, "end": 16}]},
                    {"id": 5, "spans": [{"start": 16, "end": 18}]},
                    {"id": 6, "spans": [{"start": 18, "end": 19}]},
                    {"id": 7, "spans": [{"start": 19, "end": 21}]},
                    {"id": 8, "spans": [{"start": 21, "end": 22}]},
                    {"id": 9, "spans": [{"start": 22, "end": 23}]},
                ],
            }
        )
        self.predictor._validate_tokenization(document=doc)
        # missing token id
        with self.assertRaises(ValueError):
            doc = Document.from_json(
                doc_dict={
                    "symbols": "I",
                    "tokens": [{"spans": [{"start": 0, "end": 1}]}],
                }
            )
            self.predictor._validate_tokenization(document=doc)
        # hyphen not its own token
        with self.assertRaises(ValueError):
            doc = Document.from_json(
                doc_dict={
                    "symbols": "wizard-of-oz",
                    "tokens": [{"spans": [{"start": 0, "end": 9}]}],
                }
            )
            self.predictor._validate_tokenization(document=doc)

    def test_cluster_tokens_by_whitespace(self):
        clusters = self.predictor._cluster_tokens_by_whitespace(document=self.doc)
        for token_ids in clusters:
            if len(token_ids) == 1:
                token = self.doc.tokens[token_ids[0]]
                self.assertEqual(
                    self.doc.symbols[token.spans[0].start : token.spans[0].end],
                    token.text,
                )
            else:
                tokens = [self.doc.tokens[token_id] for token_id in token_ids]
                spans = [span for token in tokens for span in token.spans]
                big_span = Span.small_spans_to_big_span(spans=spans)
                self.assertEqual(
                    self.doc.symbols[big_span.start : big_span.end],
                    "".join([token.text for token in tokens]),
                )

    def test_predict_with_whitespace(self):
        doc = Document.from_json(
            doc_dict={
                "symbols": "I am the wizard-of-oz.",
                "tokens": [
                    {"id": 0, "spans": [{"start": 0, "end": 1}]},
                    {"id": 1, "spans": [{"start": 2, "end": 4}]},
                    {"id": 2, "spans": [{"start": 5, "end": 8}]},
                    {"id": 3, "spans": [{"start": 9, "end": 15}]},
                    {"id": 4, "spans": [{"start": 15, "end": 16}]},
                    {"id": 5, "spans": [{"start": 16, "end": 18}]},
                    {"id": 6, "spans": [{"start": 18, "end": 19}]},
                    {"id": 7, "spans": [{"start": 19, "end": 21}]},
                    {"id": 8, "spans": [{"start": 21, "end": 22}]},
                    {"id": 9, "spans": [{"start": 22, "end": 23}]},
                ],
            }
        )
        (
            token_id_to_word_id,
            word_id_to_token_ids,
            word_id_to_text,
        ) = self.predictor._predict_with_whitespace(document=doc)
        self.assertDictEqual(
            token_id_to_word_id, {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}
        )
        self.assertDictEqual(
            word_id_to_token_ids, {0: [0], 1: [1], 2: [2], 3: [3, 4, 5, 6, 7, 8]}
        )
        self.assertDictEqual(
            word_id_to_text, {0: "I", 1: "am", 2: "the", 3: "wizard-of-oz."}
        )

    def test_find_hyphen_word_candidates(self):
        (
            token_id_to_word_id,
            word_id_to_token_ids,
            word_id_to_text,
        ) = self.predictor._predict_with_whitespace(document=self.doc)
        hyphen_word_candidates = self.predictor._find_hyphen_word_candidates(
            tokens=self.doc.tokens,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_token_ids=word_id_to_token_ids,
            word_id_to_text=word_id_to_text,
        )
        # verify the outputs correspond to hyphenated words
        for prefix_word_id, suffix_word_id in hyphen_word_candidates:
            prefix_word = word_id_to_text[prefix_word_id]
            suffix_word = word_id_to_text[suffix_word_id]
            self.assertTrue("-" in prefix_word + suffix_word)

    def test__predict(self):
        pos_words = [
            "wizard-of-oz",
            "moment-to-moment",
            "batch-to-batch",
            "Seven-day-old",
            "slow-to-fast",
            "HTLV-1-associated",
            "anti-E-selectin",
        ]
        neg_words = [
            "sig-nal-to-noise",
            "nonre-turn-to-zero",
            "comput-er-assisted",
            "concentra-tion-dependent",
            "ob-ject-oriented",
            "cog-nitive-behavioral",
            "deci-sion-makers",
        ]
        THRESHOLD = -1.0
        pos_results = self.predictor._predict(words=pos_words, threshold=THRESHOLD)
        self.assertEqual(len(pos_results), len(pos_words))
        self.assertTrue(all([r["is_edit"] is False for r in pos_results]))
        neg_results = self.predictor._predict(words=neg_words, threshold=THRESHOLD)
        self.assertEqual(len(neg_results), len(neg_words))
        self.assertTrue(all([r["is_edit"] is True for r in neg_results]))

    def test___predict(self):
        preds_pos = self.predictor._predict(words=self.pos_words)
        self.assertEqual(len(preds_pos), len(self.pos_words))
        tp = sum(preds_pos)
        fn = len(preds_pos) - tp

        preds_neg = self.predictor._predict(words=self.neg_words)
        self.assertEqual(len(preds_neg), len(self.neg_words))
        tn = sum(preds_neg)
        fp = len(preds_neg) - tn

        self.assertEqual(tp + fn + tn + fp, len(preds_pos) + len(preds_neg))

        p = tp / (tp + fp)
        r = tp / (tp + fn)

    def test_get_features(self):
        all_features, word_id_to_feature_ids = self.predictor._get_features(
            words=self.pos_words
        )
        self.assertEqual(len(word_id_to_feature_ids), len(self.pos_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )
        all_features, word_id_to_feature_ids = self.predictor._get_features(
            words=self.neg_words
        )
        self.assertEqual(len(word_id_to_feature_ids), len(self.neg_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )
