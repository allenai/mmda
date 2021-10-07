import argparse
import itertools
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass

from mmda.eval.s2 import get_paper_metadata

_TOKEN_FN = "{}-token.json"
_TOKEN_TYPES = ["train", "test"]
_SUBSPLITS = list(map(lambda x: str(x), range(5)))

_TITLE_LABEL = "Title"
_ABSTRACT_LABEL = "Abstract"
_RETRIEVE_LABELS = set([_TITLE_LABEL, _ABSTRACT_LABEL])


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    abstract: str


@dataclass
class PaperPage:
    paper_id: str
    page: int
    fields: dict[str, list[str]]

    def __init__(self, paper_id: str, page: int) -> None:
        self.paper_id = paper_id
        self.page = page
        self.fields = defaultdict(list)

    def push_token(self, token: str, field: str) -> None:
        self.fields[field].append(token)


def extract_title(sorted_pages: list[PaperPage]) -> str:
    title_tokens = []
    min_page = None

    for page in sorted_pages:
        if min_page and page.page != min_page:
            break
        elif len(page.fields[_TITLE_LABEL]) > 0:
            min_page = page.page

        for word in page.fields[_TITLE_LABEL]:
            title_tokens.append(word)

    return " ".join(title_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--split-path", type=str, nargs="?", required=True)
    parser.add_argument("--output-json-path", type=str, nargs="?", required=True)
    args = parser.parse_args()

    pages = []
    paper_ids = set()

    for split_id in _SUBSPLITS:
        for split in _TOKEN_TYPES:
            with open(
                os.path.join(args.split_path, split_id, _TOKEN_FN.format(split))
            ) as f:
                loaded = json.loads(f.read())

            files = loaded["files"]
            for idx, fn in enumerate(files):
                id_, page = fn.split("-")
                paper_ids.add(id_)

    # for split in _TOKEN_TYPES:
    #    with open(os.path.join(args.split_path, _TOKEN_FN.format(split))) as f:
    #        loaded = json.loads(f.read())

    #    labels = {v: k for k, v in loaded["labels"].items()}
    #    files = loaded["files"]

    #    for idx, fn in enumerate(files):
    #        id_, page = fn.split("-")

    #        word_tokens = loaded["data"][idx]["words"]
    #        word_labels = loaded["data"][idx]["labels"]

    #        assert len(word_tokens) == len(word_labels)

    #        page = PaperPage(
    #            paper_id=id_,
    #            page=int(page),
    #        )
    #        for word_idx, word in enumerate(word_tokens):
    #            page.push_token(word, labels[word_labels[word_idx]])

    #        paper_ids.add(id_)
    #        pages.append(page)

    print(len(paper_ids))

    import sys

    sys.exit()

    pages = sorted(pages, key=lambda p: p.paper_id)
    metadatas = []

    for paper_id, papers in itertools.groupby(pages, lambda p: p.paper_id):
        sorted_pages = sorted(list(papers), key=lambda p: p.page)

        # title = extract_title(sorted_pages)
        print(paper_id)

        metadatas.append(asdict(get_paper_metadata(paper_id)))

    print(len(metadatas))

    with open(args.output_json_path, "w") as f:
        f.write(json.dumps(metadatas))
        f.write("\n")
