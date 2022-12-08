"""
DictionaryWordPredictor -- Reads `tokens` and converts them into whole `words`.

Let's consider 4 consecutive rows:

    This is a few-shot learn-
    ing paper. We try few-
    shot techniques. These
    methods are useful.

This predictor tries to:
    1. merge tokens ["learn", "-", "ing"] into "learning"
    2. merge tokens ["few", "-", "shot"] into "few-shot"
    3. keep tokens ["These", "methods"] separate
    4. keep tokens ["useful", "."] separate

This technique requires 2 passes through the data:

    1. Build a dictionary of valid words, e.g. if "learning" was ever used, then we can merge
       ["learn", "-", "ing"].

    2. Classify every pair of tokens as belonging to the *same* or *different* words.

@kylel, @rauthur

"""

import os
from typing import Optional, Set, List, Tuple, Iterable, Dict

from collections import defaultdict
from mmda.parsers import PDFPlumberParser
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.predictors.heuristic_predictors.whitespace_predictor import WhitespacePredictor
from mmda.types import Metadata, Document, Span, SpanGroup
from mmda.types.names import RowsField, TokensField


class Dictionary:
    def __init__(self, words: Iterable[str], punct: str):
        self.words = set()
        self.punct = punct
        for word in words:
            self.add(word)

    def add(self, text: str) -> None:
        self.words.add(self.strip_punct(text=text.strip().lower()))

    def is_in(self, text: str, strip_punct=True) -> bool:
        if strip_punct:
            return self.strip_punct(text=text.strip().lower()) in self.words
        else:
            return text.strip().lower() in self.words

    def strip_punct(self, text: str) -> str:
        start = 0
        while start < len(text) and text[start] in self.punct:
            start += 1
        end = len(text) - 1
        while text[end] in self.punct and end > 0:
            end -= 1
        return text[start: end + 1]


class DictionaryWordPredictor(BasePredictor):
    REQUIRED_BACKENDS = None
    REQUIRED_DOCUMENT_FIELDS = [RowsField, TokensField]

    def __init__(
            self,
            dictionary_file_path: Optional[str] = None,
            punct: Optional[str] = PDFPlumberParser.DEFAULT_PUNCTUATION_CHARS
    ) -> None:
        """Build a predictor that indexes the given dictionary file.
        A dictionary is simply a case-sensitive list of words as a text file.
        Words should be lower-case in the dictionary unless they are invalid
        as a lower-case word (e.g., a person's name).

        For an example see https://github.com/dwyl/english-words (words_alpha.txt) or
        check the `tests/fixtures/example-dictionary.txt` file.

        The above example file contains base words, plurals, past-tense versions, etc.
        Thus the heuristics here do not do any changes to word structure other than
        basics:

          - Combine hyphenated words and see if they are in the dictionary
          - Strip plural endings "(s)" and punctuation

        Args:
            dictionary_file_path (str): [description]
            punct (str): [description]
        """
        self.dictionary_file_path = dictionary_file_path
        self.punct = punct
        self._dictionary = Dictionary(words=[], punct=punct)
        if self.dictionary_file_path:
            if os.path.exists(self.dictionary_file_path):
                with open(self.dictionary_file_path, 'r') as f_in:
                    for line in f_in:
                        self._dictionary.add(line)
            else:
                raise FileNotFoundError(f'{self.dictionary_file_path}')
        self.whitespace_predictor = WhitespacePredictor()

    def predict(self, document: Document) -> List[SpanGroup]:
        """Get words from a document as a list of SpanGroup.

        Args:
            document (Document): The document to process

        Raises:
            ValueError: If rows are found that do not contain any tokens

        Returns:
            list[SpanGroup]: SpanGroups where hyphenated words are joined.
                Casing and punctuation are preserved. Hyphenated words are
                only joined across row boundaries.

        Usage:
        >>> doc = # a Document with rows ("Please provide cus-", "tom models.")
        >>> predictor = DictionaryWordPredictor(dictionary_file_path="/some/file.txt")
        >>> words = predictor.predict(doc)
        >>> [w.text for w in words]
        "Please provide custom models."
        """
        self._doc_field_checker(document)

        # 1) whitespace tokenize document & compute 'adjacent' token_ids
        token_id_to_token_ids = self._precompute_whitespace_tokens(document=document)

        # 2) precompute features about each token specific to whether it's
        #    start/end of a row, or whether it corresponds to punctuation
        row_start_after_hyphen_token_ids, row_end_with_hyphen_token_ids, \
        max_row_end_token_id_to_min_row_start_token_id, punct_r_strip_candidate_token_ids, \
        punct_l_strip_candidate_token_ids = \
            self._precompute_token_features(
                document=document,
                token_id_to_token_ids=token_id_to_token_ids
            )

        # 3) build dictionary
        internal_dictionary = self._build_internal_dictionary(
            document=document,
            token_id_to_token_ids=token_id_to_token_ids,
            row_start_after_hyphen_token_ids=row_start_after_hyphen_token_ids,
            row_end_with_hyphen_token_ids=row_end_with_hyphen_token_ids
        )

        # 4) predict words for using token features
        token_id_to_word_id, word_id_to_text = self._predict_tokens(
            document=document,
            internal_dictionary=internal_dictionary,
            token_id_to_token_ids=token_id_to_token_ids,
            row_start_after_hyphen_token_ids=row_start_after_hyphen_token_ids,
            row_end_with_hyphen_token_ids=row_end_with_hyphen_token_ids,
            max_row_end_token_id_to_min_row_start_token_id=max_row_end_token_id_to_min_row_start_token_id,
            punct_r_strip_candidate_token_ids=punct_r_strip_candidate_token_ids,
            punct_l_strip_candidate_token_ids=punct_l_strip_candidate_token_ids
        )

        # 5) transformation
        words: List[SpanGroup] = self._convert_to_words(
            document=document,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_text=word_id_to_text
        )

        # 6) cleanup
        document.remove(field_name='_ws_tokens')
        return words

    def _precompute_whitespace_tokens(self, document: Document) -> Dict:
        """
        `whitespace_tokenization` is necessary because lack of whitespace is an indicator that
        adjacent tokens belong in a word together.
        """
        _ws_tokens: List[SpanGroup] = self.whitespace_predictor.predict(document=document)
        document.annotate(_ws_tokens=_ws_tokens)

        # token -> ws_tokens
        token_id_to_ws_token_id = {}
        for token in document.tokens:
            token_id_to_ws_token_id[token.id] = token._ws_tokens[0].id

        # ws_token -> tokens
        ws_token_id_to_tokens = defaultdict(list)
        for token_id, ws_token_id in token_id_to_ws_token_id.items():
            ws_token_id_to_tokens[ws_token_id].append(token_id)

        # token -> all cluster tokens
        token_id_to_token_ids = {}
        for token_id, ws_token_id in token_id_to_ws_token_id.items():
            candidate_token_ids = [i for i in ws_token_id_to_tokens[ws_token_id]]
            token_id_to_token_ids[token_id] = candidate_token_ids

        return token_id_to_token_ids

    def _precompute_token_features(
            self,
            document: Document,
            token_id_to_token_ids
    ) -> Tuple:
        """
        Compute stuff necessary for dictionary-building and/or merging tokens into words.

        1. `beginning|end_of_row|page` is necessary because row transitions are often where tokens
           should be merged into a word. Knowing this also helps with determining what are "safe"
           words to add to dictionary.

        2. `punctuation` in `start_of_row` tokens is necessary because we may need to keep them
            as separate tokens even if there is a word merge (e.g. the semicolon "fine-tuning;")
        """

        # beginning/end of row w/ hyphen
        # TODO: add pages too
        row_end_with_hyphen_token_ids = set()
        row_start_after_hyphen_token_ids = set()
        max_row_end_token_id_to_min_row_start_token_id = {}
        for i in range(0, len(document.tokens) - 1):
            current = document.tokens[i]
            next = document.tokens[i + 1]
            is_transition = current.rows[0].id != next.rows[0].id
            has_hyphen = current._ws_tokens[0].text.endswith('-')
            has_prefix = current._ws_tokens[0].text != '-'  # avoids cases where "-" by itself
            if is_transition and has_hyphen and has_prefix:
                row_end_token_ids = sorted([token.id for token in current._ws_tokens[0].tokens])
                row_start_token_ids = sorted([token.id for token in next._ws_tokens[0].tokens])
                for i in row_end_token_ids:
                    row_end_with_hyphen_token_ids.add(i)
                for j in row_start_token_ids:
                    row_start_after_hyphen_token_ids.add(j)
                max_row_end_token_id_to_min_row_start_token_id[
                    max(row_end_token_ids)
                ] = min(row_start_token_ids)

        # also, keep track of potential punct token_ids to right-strip (e.g. ',' in 'models,')
        # should apply to all tokens except those at end of a row
        punct_r_strip_candidate_token_ids = set()
        for token in document.tokens:
            candidate_token_ids = token_id_to_token_ids[token.id]
            if len(candidate_token_ids) > 1:
                # r-strip logic. keep checking trailing tokens for punct; stop as soon as not
                if token.id not in row_end_with_hyphen_token_ids:
                    for k in candidate_token_ids[::-1]:
                        if document.tokens[k].text in self._dictionary.punct:
                            punct_r_strip_candidate_token_ids.add(k)
                        else:
                            break

        # also track of potential punct token_ids to left-strip (e.g. '(' in '(BERT)')
        # should apply to all tokens.
        # avoid tracking cases where it's all punctuation (e.g. '...')
        punct_l_strip_candidate_token_ids = set()
        for token in document.tokens:
            candidate_token_ids = token_id_to_token_ids[token.id]
            is_multiple_tokens_wout_whitespace = len(candidate_token_ids) > 1
            is_entirely_punctuation = all([
                document.tokens[i].text in self._dictionary.punct for i in candidate_token_ids
            ])
            if is_multiple_tokens_wout_whitespace and not is_entirely_punctuation:
                # l-strip logic. keep checking prefix tokens for punct; stop as soon as not
                for k in candidate_token_ids:
                    if document.tokens[k].text in self._dictionary.punct:
                        punct_l_strip_candidate_token_ids.add(k)
                    else:
                        break

        return row_start_after_hyphen_token_ids, \
               row_end_with_hyphen_token_ids, \
               max_row_end_token_id_to_min_row_start_token_id, \
               punct_r_strip_candidate_token_ids, \
               punct_l_strip_candidate_token_ids

    def _build_internal_dictionary(
            self,
            document: Document,
            token_id_to_token_ids,
            row_start_after_hyphen_token_ids,
            row_end_with_hyphen_token_ids
    ) -> Dictionary:
        """dictionary of possible words"""
        internal_dictionary = Dictionary(words=self._dictionary.words, punct=self.punct)
        for token in document.tokens:
            if token.id in row_end_with_hyphen_token_ids:
                continue
            if token.id in row_start_after_hyphen_token_ids:
                continue
            candidate_text = ''.join(
                [document.tokens[i].text for i in token_id_to_token_ids[token.id]]
            )
            internal_dictionary.add(candidate_text)
        return internal_dictionary

    def _predict_tokens(
            self,
            document: Document,
            internal_dictionary: Dictionary,
            token_id_to_token_ids,
            row_start_after_hyphen_token_ids,
            row_end_with_hyphen_token_ids,
            max_row_end_token_id_to_min_row_start_token_id,
            punct_r_strip_candidate_token_ids,
            punct_l_strip_candidate_token_ids
    ) -> Tuple[Dict, Dict]:

        token_id_to_word_id = {token.id: None for token in document.tokens}
        word_id_to_token_ids = defaultdict(list)
        word_id_to_text = {}

        # easy case first! most words aren't split & are their own tokens
        for token in document.tokens:
            clustered_token_ids = token_id_to_token_ids[token.id]
            if (
                            not token.id in row_end_with_hyphen_token_ids and
                            not token.id in row_start_after_hyphen_token_ids and
                            len(clustered_token_ids) == 1
            ):
                token_id_to_word_id[token.id] = token.id
                word_id_to_token_ids[token.id].append(token.id)
                word_id_to_text[token.id] = token.text
            else:
                pass

        # loop through remaining tokens. start with ones without row split, as that's easier.
        for token in document.tokens:
            if (
                            not token.id in row_end_with_hyphen_token_ids and
                            not token.id in row_start_after_hyphen_token_ids and
                            token_id_to_word_id[token.id] is None
            ):
                # calculate 2 versions of the text to check against dictionary:
                # one version is raw concatenate all adjacent tokens
                # another version right-strips punctuation after concatenating
                clustered_token_ids = token_id_to_token_ids[token.id]
                first_token_id = min(clustered_token_ids)
                candidate_text = ''.join([document.tokens[i].text for i in clustered_token_ids])
                clustered_token_ids_r_strip_punct = [
                    i for i in clustered_token_ids
                    if i not in punct_r_strip_candidate_token_ids
                ]
                candidate_text_strip_punct = ''.join([
                    document.tokens[i].text for i in clustered_token_ids_r_strip_punct
                ])

                # if concatenated tokens are in dictionary as-is, take them as a single word
                if internal_dictionary.is_in(text=candidate_text, strip_punct=False):
                    for i in clustered_token_ids:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text
                # otherwise, default is to take all adjacent tokens (w/out whitespace between them)
                # as a single word & strip punctuation
                else:
                    for i in clustered_token_ids_r_strip_punct:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text_strip_punct
                    for i in clustered_token_ids:
                        if i in punct_r_strip_candidate_token_ids:
                            token_id_to_word_id[i] = i
                            word_id_to_token_ids[i].append(i)
                            word_id_to_text[i] = document.tokens[i].text
            else:
                pass

        # finally, handle tokens that are split across rows
        for token in document.tokens:
            if (
                            token.id in max_row_end_token_id_to_min_row_start_token_id and
                            token_id_to_word_id[token.id] is None
            ):
                # calculate 4 versions of the text to check against dictionary:
                #  1. one version is raw concatenate all adjacent tokens
                #  2. another version right-strips punctuation after concatenating
                #  3. you can also do a raw concatenation *after* removing bridging '-' character
                #  4. and you can remove the '-' as well as perform a right-strip of punctuation
                start_token_ids = token_id_to_token_ids[token.id]
                first_token_id = min(start_token_ids)
                end_token_ids = token_id_to_token_ids[
                    max_row_end_token_id_to_min_row_start_token_id[token.id]
                ]
                end_token_ids_strip_punct = [
                    i for i in end_token_ids if i not in punct_r_strip_candidate_token_ids
                ]
                start_text = ''.join([document.tokens[i].text for i in start_token_ids])
                end_text = ''.join([document.tokens[i].text for i in end_token_ids])
                end_text_strip_punct = ''.join([
                    document.tokens[i].text for i in end_token_ids_strip_punct
                ])
                assert start_text[-1] == '-'

                candidate_text = start_text + end_text
                candidate_text_strip_punct = start_text + end_text_strip_punct
                candidate_text_no_hyphen = start_text[:-1] + end_text
                candidate_text_strip_punct_no_hyphen = start_text[:-1] + end_text_strip_punct

                # if concatenated tokens are in dictionary as-is, take them as a single word
                if internal_dictionary.is_in(text=candidate_text, strip_punct=False):
                    for i in start_token_ids + end_token_ids:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text
                # if concatenated tokens wout hyphen are in dictionary
                elif internal_dictionary.is_in(text=candidate_text_no_hyphen, strip_punct=False):
                    for i in start_token_ids + end_token_ids:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text_no_hyphen
                # if concatenated tokens wout hyphen *AND* right-strip punct are in dict..
                elif internal_dictionary.is_in(text=candidate_text_strip_punct_no_hyphen,
                                               strip_punct=False):
                    for i in start_token_ids + end_token_ids_strip_punct:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text_strip_punct_no_hyphen
                    for i in end_token_ids:
                        if i in punct_r_strip_candidate_token_ids:
                            token_id_to_word_id[i] = i
                            word_id_to_token_ids[i].append(i)
                            word_id_to_text[i] = document.tokens[i].text
                # if concatenated tokens are *NOT* in dictionary, default behavior is
                # to concatenate anyways, keeping hyphen, and stripping punctuation as
                # separate tokens
                else:
                    for i in start_token_ids + end_token_ids_strip_punct:
                        token_id_to_word_id[i] = first_token_id
                        word_id_to_token_ids[first_token_id].append(i)
                    word_id_to_text[first_token_id] = candidate_text_strip_punct
                    for i in end_token_ids:
                        if i in punct_r_strip_candidate_token_ids:
                            token_id_to_word_id[i] = i
                            word_id_to_token_ids[i].append(i)
                            word_id_to_text[i] = document.tokens[i].text
            else:
                pass

        # 2022-12; we need to handle cases like '(CS)' where the word starts with a punct "("
        # we actually do want this as a separate word. but under the above logic, we would've
        # assigned "(" and "CS" the same word ID.
        # this part identifies words that begin with punctuation, and splits them.
        for token in document.tokens:
            if token.id in punct_l_strip_candidate_token_ids:
                # capture current state, before fixing
                word_id = token_id_to_word_id[token.id]
                word_text = word_id_to_text[word_id]
                other_same_word_token_ids = [
                    i for i in word_id_to_token_ids[token_id_to_word_id[token.id]]
                    if token_id_to_word_id[i] == word_id and i != token.id
                ]
                new_first_token_id = min(other_same_word_token_ids)
                # update this punctuation token to be its own word
                word_id_to_text[word_id] = token.text
                word_id_to_token_ids[word_id] = [token.id]
                # update subsequent tokens that were implicated. fix their word_id. fix the text.
                for other_token_id in other_same_word_token_ids:
                    token_id_to_word_id[other_token_id] = new_first_token_id
                word_id_to_token_ids[new_first_token_id] = other_same_word_token_ids
                word_id_to_text[new_first_token_id] = word_text.lstrip(token.text)
        # this data structure has served its purpose by this point
        del word_id_to_token_ids

        # edge case handling. there are cases (e.g. tables) where each cell is detected as its own
        # row. This is super annoying but *shrug*. In these cases, a cell "-" followed by another
        # cell "48.9" can be represented as 2 adjacent rows. This can cause the token for "48"
        # to be implicated in `row_start_after_hyphen_token_ids`, and thus the tokens "48.9"
        # wouldn't be processed under the Second module above... But it also wouldnt be processed
        # under the Third module above because the preceding hyphen "-" wouldn't have made it to
        # `row_end_with_hyphen_token_ids` (as it's by itself).
        # Anyways... this case is soooooo specific that for now, the easiest thing is to just
        # do a final layer of passing over unclassified tokens & assigning word_id based on
        # whitespace.
        #
        # 2022-12: actually this may not be needed anymore after modifying featurization function
        # to avoid considering cases where a row only contains a single hyphen.
        # commented out for now.
        #
        # for token in document.tokens:
        #     if token_id_to_word_id[token.id] is None:
        #         clustered_token_ids = token_id_to_token_ids[token.id]
        #         first_token_id = min(clustered_token_ids)
        #         candidate_text = ''.join([document.tokens[i].text for i in clustered_token_ids])
        #         for i in clustered_token_ids:
        #             token_id_to_word_id[i] = first_token_id
        #         word_id_to_text[first_token_id] = candidate_text

        # are there any unclassified tokens?
        assert None not in token_id_to_word_id.values()
        return token_id_to_word_id, word_id_to_text

    def _convert_to_words(
            self,
            document: Document,
            token_id_to_word_id,
            word_id_to_text
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
                    metadata=Metadata(text=word_id_to_text[current_word_id])
                )
                words.append(word)
                tokens_in_word = [token]
                current_word_id = word_id
        # last bit
        word = SpanGroup(
            spans=[span for token in tokens_in_word for span in token.spans],
            id=current_word_id,
            metadata=Metadata(text=word_id_to_text[current_word_id])
        )
        words.append(word)
        return words
