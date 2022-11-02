import pandas as pd
from pydantic import BaseModel
import re
from thefuzz import fuzz
from typing import List, Tuple, Dict

from mmda.types.annotation import SpanGroup


DIGITS = re.compile(r'[0-9]+')
ALPHA = re.compile(r'[A-Za-z]+')
RELEVANT_PUNCTUATION = re.compile(r"\(|\)|\[|,|\]|\.|&|\;")

FUZZ_RATIO = "fuzz_ratio"
JACCARD_1GRAM = "jaccard_1gram"
JACCARD_2GRAM = "jaccard_2gram"
JACCARD_3GRAM = "jaccard_3gram"
JACCARD_4GRAM = "jaccard_4gram"
HEURISTIC = "heuristic"
HAS_SOURCE_TEXT = "has_source_text"
JACCARD_NUMERIC = "jaccard_numeric"
MATCH_NUMERIC = "match_numeric"
JACCARD_ALPHA = "jaccard_alpha"
MATCH_FIRST_TOKEN = "match_first_token"

class CitationLink:
    def __init__(self, mention: SpanGroup, bib: SpanGroup):
        self.mention = mention
        self.bib = bib

    def to_text_dict(self) -> Dict[str, str]:
        return {"source_text": "".join(self.mention.symbols), "target_text": "".join(self.bib.symbols)}

def featurize(possible_links: List[CitationLink]) -> pd.DataFrame:
    # create dataframe
    df = pd.DataFrame.from_records([link.to_text_dict() for link in possible_links])

    df[FUZZ_RATIO] = df.apply(lambda row: fuzz.ratio(row['source_text'], row['target_text']), axis=1)
    df[JACCARD_1GRAM] = df.apply(lambda row: jaccardify(row['source_text'], row['target_text'], 1), axis=1)
    df[JACCARD_2GRAM] = df.apply(lambda row: jaccardify(row['source_text'], row['target_text'], 2), axis=1)
    df[JACCARD_3GRAM] = df.apply(lambda row: jaccardify(row['source_text'], row['target_text'], 3), axis=1)
    df[JACCARD_4GRAM] = df.apply(lambda row: jaccardify(row['source_text'], row['target_text'], 4), axis=1)
    df[HEURISTIC] = df.apply(lambda row: match_source_tokens(row['source_text'], row['target_text']), axis=1)
    df[HAS_SOURCE_TEXT] = df.apply(lambda row: has_source_text(row['source_text']), axis=1)
    df[JACCARD_NUMERIC] = df.apply(lambda row: jaccard_numeric(row['source_text'], row['target_text']), axis=1)
    df[MATCH_NUMERIC] = df.apply(lambda row: match_numeric(row['source_text'], row['target_text']), axis=1)
    df[JACCARD_ALPHA] = df.apply(lambda row: jaccard_alpha(row['source_text'], row['target_text']), axis=1)
    df[MATCH_FIRST_TOKEN] = df.apply(lambda row: match_first_token(row['source_text'], row['target_text']), axis=1)
    
    # drop text columns
    X_features = df.drop(columns=['source_text', 'target_text'])
    return X_features


def ngramify(s: str, n: int) -> List[str]:
    s_len = len(s)
    return [s[i:i+n] for i in range(s_len-n+1)]

def jaccard_ngram(ngrams1: List[str], ngrams2: List[str]) -> float:
    if ngrams1 or ngrams2:
        s1 = set(ngrams1)
        s2 = set(ngrams2)
        return len(s1.intersection(s2)) / len(s1.union(s2))
    else:
        return 0.0

def jaccardify(source: str, target: str, n: int) -> float:
    truncated_target = target[:50]
    source_ngrams = ngramify(source, n)
    target_ngrams = ngramify(truncated_target, n)
    return jaccard_ngram(source_ngrams, target_ngrams)

def has_source_text(source: str) -> int:
    if source.strip():
        return 1
    else:
        return 0

def jaccard_numeric(source: str, target: str) -> float:
    source_numerics = re.findall(DIGITS, source)
    truncated_target = target[:100]
    target_numerics = re.findall(DIGITS, truncated_target)
    return jaccard_ngram(source_numerics, target_numerics)

def match_numeric(source: str, target: str) -> float:
    source_numerics = re.findall(DIGITS, source)
    truncated_target = target[:100]
    target_numerics = re.findall(DIGITS, truncated_target)
    token_found = []
    for number in source_numerics:
        found = number in target_numerics
        token_found.append(found)
    
    if False not in token_found:
        return 1
    else:
        return 0

def jaccard_alpha(source: str, target: str) -> float:
    source_alpha = re.findall(ALPHA, source)
    truncated_target = target[:50]
    target_alpha = re.findall(ALPHA, truncated_target)
    return jaccard_ngram(source_alpha, target_alpha)

# predicts mention/bib entry matches by matching normalized source tokens

# examples returning True:
# source_text = "[78]"
# target_text = "[78]. C. L. Willis and S. L. Miertschin. Mind maps..."
# source_text = "(Wilkinson et al., 2017)"
# target_text = "Wilkinson, R., Quigley, Q., and Marimba, P. Time means nothing. Journal of Far-Fetched Hypotheses, 2017."
#
# examples returning False:
# source_text = "[3]"
# target_text = "[38] J. Koch, A. Lucero, L. Hegemann, and A. Oulas..."
# source_text = "(Shi 2020)"
# target_text = "Shi, X. Vanilla Ice Cream Is the Best. Epic Assertions, 2021"

# some failure modes: no source text; source text ranges such as "[13-15]";
# incomplete source text such as ", 2019)"; bib entry text with both item and page numbers
def strip_and_tokenize(text: str) -> List[str]:
    stripped_text = RELEVANT_PUNCTUATION.sub("", text)
    return stripped_text.lower().strip().split()

def match_source_tokens(source: str, target: str) -> float:
    if not source:
        return 0
    else:
        source_tokens = strip_and_tokenize(source)
        target_tokens = strip_and_tokenize(target)
        token_found = []
        for token in source_tokens:
            if token != 'et' and token != 'al' and token != 'and':
                found = token in target_tokens
                token_found.append(found)
        
        if False not in token_found:
            return 1
        else:
            return 0


def match_first_token(source: str, target: str) -> float:
    truncated_target = target[:50]
    source_tokens = strip_and_tokenize(source)
    target_tokens = strip_and_tokenize(truncated_target)

    if not source_tokens:
        return 0
    else:
        first_source_token = source_tokens[0]
        if first_source_token in target_tokens:
            return 1
        else:
            return 0