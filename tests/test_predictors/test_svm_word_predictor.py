"""
Tests for SVM Word Predictor

@kylel
"""

import json
import os
import unittest
from typing import List, Optional, Set

import numpy as np

from mmda.predictors.heuristic_predictors.svm_word_predictor import (
    SVMClassifier,
    SVMWordPredictor,
)
from mmda.types import Document, Span, SpanGroup


class TestSVMClassifier(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.fixture_path = os.path.join(os.path.dirname(__file__), "../fixtures/")
        self.classifier = SVMClassifier.from_path(
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

    def test_batch_predict_unit(self):
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
        THRESHOLD = -1.5
        pos_results = self.classifier.batch_predict(
            words=pos_words, threshold=THRESHOLD
        )
        self.assertEqual(len(pos_results), len(pos_words))
        self.assertTrue(all([r.is_edit is False for r in pos_results]))
        neg_results = self.classifier.batch_predict(
            words=neg_words, threshold=THRESHOLD
        )
        self.assertEqual(len(neg_results), len(neg_words))
        self.assertTrue(all([r.is_edit is True for r in neg_results]))

    def test_batch_predict_eval(self):
        """
        As a guideline, we want Recall to be close to 1.0 because we want
        the model to favor predicting things as "negative" (i.e. not an edit).
        If the classifier predicts a "1", then essentially we don't do anything.
        Meaning in all cases where the ground truth is "1" (dont do anything),
        we want to recover all these cases nearly perfectly, and ONLY
        take action when absolutely safe.

        THRESHOLD = -1.7    --> P: 0.9621262458471761 R: 1.0
        THRESHOLD = -1.6    --> P: 0.9674346429879954 R: 1.0
        THRESHOLD = -1.5    --> P: 0.9716437941036409 R: 1.0
        THRESHOLD = -1.4    --> P: 0.9755705281460552 R: 0.9999554446622705
        THRESHOLD = -1.0    --> P: 0.9866772193641999 R: 0.9998217786490822
        THRESHOLD = -0.5    --> P: 0.9955352184633155 R: 0.9984405631794689
        THRESHOLD = 0.0     --> P: 0.9985657299090135 R: 0.9926483692746391
        THRESHOLD = 1.0     --> P: 0.9997759019944723 R: 0.8944929602566387
        """
        THRESHOLD = -1.5
        preds_pos = self.classifier.batch_predict(
            words=self.pos_words, threshold=THRESHOLD
        )
        self.assertEqual(len(preds_pos), len(self.pos_words))
        preds_pos_as_ints = [int(r.is_edit is False) for r in preds_pos]
        tp = sum(preds_pos_as_ints)
        fn = len(preds_pos_as_ints) - tp

        preds_neg = self.classifier.batch_predict(
            words=self.neg_words, threshold=THRESHOLD
        )
        self.assertEqual(len(preds_neg), len(self.neg_words))
        preds_neg_as_ints = [int(r.is_edit is True) for r in preds_neg]
        tn = sum(preds_neg_as_ints)
        fp = len(preds_neg_as_ints) - tn

        self.assertEqual(tp + fn + tn + fp, len(preds_pos) + len(preds_neg))

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        # uncomment for debugging
        print(f"P: {p} R: {r}")

        self.assertGreaterEqual(p, 0.9)
        self.assertGreaterEqual(r, 0.9)

    def test_get_features(self):
        (
            all_features,
            word_id_to_feature_ids,
        ) = self.classifier._get_features(words=self.pos_words)
        self.assertEqual(len(word_id_to_feature_ids), len(self.pos_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )
        (
            all_features,
            word_id_to_feature_ids,
        ) = self.classifier._get_features(words=self.neg_words)
        self.assertEqual(len(word_id_to_feature_ids), len(self.neg_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )


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
            os.path.join(
                self.fixture_path,
                "types/20fdafb68d0e69d193527a9a1cbe64e7e69a3798__pdfplumber_doc.json",
            )
        ) as f_in:
            doc_dict = json.load(f_in)
            self.doc = Document.from_json(doc_dict=doc_dict)

    def test_predict_no_hyphen(self):
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
                ],
            }
        )
        words = self.predictor.predict(document=doc)
        self.assertEqual(len(words), 4)
        doc.annotate(words=words)
        self.assertListEqual(
            [w.text for w in words], ["I", "am", "the", "wizard-of-oz."]
        )

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
        # uncomment for debugging:
        for word in self.doc.words:
            token_text = str([t.text for t in word.tokens])
            if len(word.tokens) > 1 and "-" in token_text:
                print(f"{token_text}\t-->\t{word.text}")

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
