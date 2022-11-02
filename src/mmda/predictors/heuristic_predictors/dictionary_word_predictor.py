"""
DictionaryWordPredictor -- Reads rows of text into dehyphenated words.

@rauthur
"""

import string
from typing import Optional, Set, List

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.metadata import Metadata
from mmda.types.annotation import Annotation, Span, SpanGroup
from mmda.types.document import Document
from mmda.types.names import Rows, Tokens


class DictionaryWordPredictor(BasePredictor):

    REQUIRED_BACKENDS = None
    REQUIRED_DOCUMENT_FIELDS = [Rows, Tokens]

    _dictionary: Optional[Set[str]] = None

    def __init__(self, dictionary_file_path: str) -> None:
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
        """
        self.dictionary_file_path = dictionary_file_path

    @property
    def dictionary(self) -> Set[str]:
        """Global dictionary and not document specific. This dictionary is the basis for
        finding words and will be appended with document-level tokens.

        Returns:
            set[str]: A set of words in the dictionary. For proper names, the dictionary
            may contain just the title-cased word (e.g., "Russell" and not "russell").
        """
        if not self._dictionary:
            # TODO: Sanity checks for dictionary read results
            with open(self.dictionary_file_path, "r") as source:
                self._dictionary = set(source.read().split())

        return self._dictionary

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
        words = []
        local_dictionary = self._build_document_local_dictionary(document)

        skip_first_token = False

        for curr_row, next_row in self._row_pairs(document=document):

            # Skip first token and there is only one token -> skip entire row
            if skip_first_token and len(curr_row.tokens) == 1:
                skip_first_token = False
                continue

            # Add all tokens except last to document words
            for i, token in enumerate(curr_row.tokens[:-1]):
                if skip_first_token and i == 0:
                    skip_first_token = False
                    continue

                words.append(self._copy_token_with_text(token))

            if len(curr_row.tokens) == 0:
                raise ValueError("Encountered row without any tokens!")

            curr_row_last_token = curr_row.tokens[-1]

            # If we are at the end of the document just add token
            if not next_row:
                words.append(self._copy_token_with_text(curr_row_last_token))
                continue

            curr_row_last_token_text = self._token_text(curr_row_last_token)

            # If last token is unhyphenated just add token
            if not curr_row_last_token_text.endswith("-"):
                words.append(self._copy_token_with_text(curr_row_last_token))
                continue

            # Otherwise see if we should join to start of next row
            next_row_first_token = next_row.tokens[0]
            next_row_first_token_text = self._token_text(next_row_first_token)

            # Remove dangling punctutation like commas for word detection
            if next_row_first_token_text[-1] in string.punctuation:
                next_row_first_token_text = next_row_first_token_text[:-1]

            # Remove optional pluralization at end of token
            for plural_suffix in ["(s)", "(s"]:
                if next_row_first_token_text[-len(plural_suffix) :] == plural_suffix:
                    next_row_first_token_text = next_row_first_token_text[
                        : -len(plural_suffix)
                    ]

            # Combined word is in dictionary without hyphen (JOIN)
            combined_no_hyphen = "".join(
                [curr_row_last_token_text[:-1], next_row_first_token_text]
            )

            # When processing the next line, skip the first token
            skip_first_token = True

            # Restore original text without any punctuation stripping
            if (
                combined_no_hyphen in self.dictionary
                or combined_no_hyphen.lower() in self.dictionary
                or combined_no_hyphen.lower() in local_dictionary
            ):
                combined_text = curr_row_last_token_text[:-1] + \
                                self._token_text(next_row_first_token)
            else:
                # Use the combined, hyphenated word instead (e.g., few-shot)
                combined_text = curr_row_last_token_text + \
                                self._token_text(next_row_first_token)
            span_group = SpanGroup(
                spans=curr_row_last_token.spans + next_row_first_token.spans
            )
            span_group.text = combined_text
            words.append(span_group)

        # add IDs to each word
        for i, word in enumerate(words):
            word.id = i

        return words

    def _token_text(self, token: SpanGroup) -> str:
        return "".join(token.symbols)

    def _copy_token_with_text(self, token: SpanGroup) -> SpanGroup:
        sg = SpanGroup(spans=token.spans, metadata=Metadata(text=self._token_text(token)))
        return sg

    def _row_pairs(self, document):
        for i in range(0, len(document.rows) - 1):
            yield document.rows[i], document.rows[i + 1]

        # Consume from current row so don't skip last
        yield document.rows[-1], None

    def _build_document_local_dictionary(self, document: Document) -> Set[str]:
        puncset = set(string.punctuation)
        local_dictionary = set()

        for symbol_group in document.symbols.split():
            # Toss out anything with punctutation
            if len(puncset & set(symbol_group)) > 0:
                continue

            local_dictionary.add(symbol_group.lower())

        return local_dictionary
