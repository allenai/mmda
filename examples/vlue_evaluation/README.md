# Evaluate a predictor against the VLUE dataset

## Current results

In the tables below S2 uses metadata retrieved directly from the [Semantic Scholar API](https://api.semanticscholar.org/graph/v1#operation/get_graph_get_paper) without any changes to title or abstract. Grobid uses version 0.7.0 running locally and the processHeaderDocument API method, see [Grobid service docs](https://grobid.readthedocs.io/en/latest/Grobid-service/).

### Title

Model                                   | Parser     | N  | Mean  | Std
--------------------------------------- | ---------- | -- | ----- | -----
S2                                      | -          | 16 | 0.920 | 0.210
Grobid                                  | -          | 16 | 0.870 | 0.300
hvila-block-layoutlm-finetuned-docbank  | SScraper   | 16 | 0.406 | 0.437
hvila-block-layoutlm-finetuned-grotoap2 | SScraper   | 16 | 0.552 | 0.432
hvila-row-layoutlm-finetuned-docbank    | SScraper   | 16 | 0.466 | 0.379
hvila-row-layoutlm-finetuned-grotoap2   | SScraper   | 16 | 0.584 | 0.408
ivila-block-layoutlm-finetuned-docbank  | SScraper   | 16 | 0.658 | 0.430
ivila-block-layoutlm-finetuned-grotoap2 | SScraper   | 16 | 0.703 | 0.356
hvila-block-layoutlm-finetuned-docbank  | PDFPlumber | 16 | 0.446 | 0.449
hvila-block-layoutlm-finetuned-grotoap2 | PDFPlumber | 16 | 0.585 | 0.448
hvila-row-layoutlm-finetuned-docbank    | PDFPlumber | 16 | 0.637 | 0.409
hvila-row-layoutlm-finetuned-grotoap2   | PDFPlumber | 16 | 0.851 | 0.295
ivila-block-layoutlm-finetuned-docbank  | PDFPlumber | 16 | 0.614 | 0.412
ivila-block-layoutlm-finetuned-grotoap2 | PDFPlumber | 16 | 0.680 | 0.358


### Abstract

Model                                   | Parser     | N  | Mean  | Std
--------------------------------------- | ---------- | -- | ----- | -----
S2                                      | -          | 16 | 0.823 | 0.317
Grobid                                  | -          | 16 | 0.816 | 0.377
hvila-block-layoutlm-finetuned-docbank  | SScraper   | 16 | 0.618 | 0.409
hvila-block-layoutlm-finetuned-grotoap2 | SScraper   | 16 | 0.832 | 0.254
hvila-row-layoutlm-finetuned-docbank    | SScraper   | 16 | 0.811 | 0.210
hvila-row-layoutlm-finetuned-grotoap2   | SScraper   | 16 | 0.814 | 0.299
ivila-block-layoutlm-finetuned-docbank  | SScraper   | 16 | 0.604 | 0.328
ivila-block-layoutlm-finetuned-grotoap2 | SScraper   | 16 | 0.744 | 0.306
hvila-block-layoutlm-finetuned-docbank  | PDFPlumber | 16 | 0.865 | 0.181
hvila-block-layoutlm-finetuned-grotoap2 | PDFPlumber | 16 | 0.841 | 0.286
hvila-row-layoutlm-finetuned-docbank    | PDFPlumber | 16 | 0.829 | 0.191
hvila-row-layoutlm-finetuned-grotoap2   | PDFPlumber | 16 | 0.970 | 0.026
ivila-block-layoutlm-finetuned-docbank  | PDFPlumber | 16 | 0.690 | 0.252
ivila-block-layoutlm-finetuned-grotoap2 | PDFPlumber | 16 | 0.775 | 0.289


These scores are based on 16 of the documents excluding the following SHAs which fail for PDF Plumber:

```
396fb2b6ec96ff74e22ddd2484a9728257cccfbf
3ef6e51baee01b4c90c188a964f2298b7c309b07
4277d1ec41d88d595a0d80e4ab4146d8c2db2539
564a73c07436e1bd75e31b54825d2ba8e4fb68b7
```

Also excluded are these SHAs which fail on SymbolScraper (failing SHAs above already removed):

```
25b3966066bfe9d17dfa2384efd57085f0c546a5
9b69f0ca8bbc617bb48d76f73d269af5230b1a5e
```

## Dataset labels

Labels used in the evaluation here have the following format:

```json
{
  "id": "semantic-scholar-paper-hash",
  "url": "URL to find paper on semanticscholar.org",
  "title": "Paper title in UTF-8",
  "abstract": "Paper abstract in UTF-8"
}
```

The labels are hand-curated and attempt to be the best possible output for a paper given just the PDF and no external metadata about the paper, authors, etc. This generally means that:

1. Characters use a Unicode representation when possible. An example of this is the superscript "+" used when referring to a positively charged ion. For example "Na+" vs. "Na⁺". The latter is used when it visually matches what is shown on the PDF regardless of any underlying character stream.

2. When multiple languages are available in the paper the English language summary is provided only. A future version of this dataset may provide additional languages.

3. Words are dehyphenated. The task here is a reconstruction of the full text without visual breaks (like multi-line hyphenation). In other words, the output of a prediction should be directly consumable by another system that expects fully reconstructed text (e.g., a screen reader, classifier, etc.).

4. Multi-paragraph abstracts are joined by a single newline character between paragraphs.

## Obtaining the labels

Currently, labels are kept on (AI2-private) AWS:

```bash
aws s3 cp s3://ai2-s2-russellr/datasets/mmda/vlue-curated-labels.json
```

This location will change as the dataset is completed and more documents are added. Currently, only 22 documents are available.

## Metrics

Currently there is a single metric provided:

```python
def score(label: str, prediction: str) -> float:
  return 1 - (levenshtein(label, prediction) / max(len(label), len(prediction)))
```

See this [Wikipedia article](https://en.wikipedia.org/wiki/Levenshtein_distance) for more information.

## Running the evaluation

For a single VILA model run use the following (using `main.py` in this folder):

```bash
python main.py \
  --pdfs-basedir ./resources/the-pdfs \
  --labels-json-path ./resources/curated-labels.json \
  --vila-type ivila \
  --vila-model-path ./resources/model/the-parameters
```

Find model weights in the [VILA repository README](https://github.com/allenai/VILA#model-weights). Find labels following the steps above in this README. Find PDFs following the steps in [this README](https://github.com/allenai/scienceparseplus-annotation) (currently AI2 private).

### Custom predictors

See the implementation in `main.py` for VILA as an example. To run any MMDA predictor some code is necessary to take the document prediction and stitch this together into a final title and abstract text. No assumptions are made about where the title/abstract data may be stored on a predicted output since MMDA does not have a fixed prediction format.

Specifically, a prediction passed to the `mmda.eval.vlue.score` function should implement the following protocol:

```python
class PredictedDoc(Protocol):
    @property
    def title(self) -> str:
      pass

    @property
    def abstract(self) -> str:
      pass
```

This protocol is defined in `mmda.eval.vlue`.

### Obtaining Grobid and S2 predictions

Functions are provided to obtain Grobid and S2 predictions in the correct format. For Grobid, an instantiated parser is expected (`GrobidHeaderParser`). For S2, nothing is needed other than calling the function. Import and use the functions `grobid_prediction` and `s2_prediction` from the `mmda.eval.vlue` module if desired.
