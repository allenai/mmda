# Evaluate a predictor against the VLUE dataset

## Current results

In the tables below S2 uses metadata retrieved directly from the [Semantic Scholar API](https://api.semanticscholar.org/graph/v1#operation/get_graph_get_paper) without any changes to title or abstract. Grobid uses version 0.7.0 running locally and the processHeaderDocument API method, see [Grobid service docs](https://grobid.readthedocs.io/en/latest/Grobid-service/).

### Title

Model                                   | N  | Mean  | Std
--------------------------------------- | -- | ----- | -----
S2                                      | 18 | 0.882 | 0.267
Grobid                                  | 18 | 0.782 | 0.382
ivila-block-layoutlm-finetuned-docbank  | 18 | 0.614 | 0.427
ivila-block-layoutlm-finetuned-grotoap2 | 18 | 0.658 | 0.381
hvila-block-layoutlm-finetuned-docbank  | 18 | 0.451 | 0.455
hvila-row-layoutlm-finetuned-docbank    | 18 | 0.639 | 0.384
hvila-block-layoutlm-finetuned-grotoap2 | 18 | 0.575 | 0.455
hvila-row-layoutlm-finetuned-grotoap2   | 18 | 0.808 | 0.343

### Abstract

Model                                   | N  | Mean  | Std
--------------------------------------- | -- | ----- | -----
S2                                      | 18 | 0.787 | 0.359
Grobid                                  | 18 | 0.780 | 0.405
ivila-block-layoutlm-finetuned-docbank  | 18 | 0.671 | 0.293
ivila-block-layoutlm-finetuned-grotoap2 | 18 | 0.725 | 0.327
hvila-block-layoutlm-finetuned-docbank  | 18 | 0.806 | 0.267
hvila-row-layoutlm-finetuned-docbank    | 18 | 0.777 | 0.269
hvila-block-layoutlm-finetuned-grotoap2 | 18 | 0.793 | 0.344
hvila-row-layoutlm-finetuned-grotoap2   | 18 | 0.908 | 0.231



```
-------- TITLE --------
grobid---
N: 18; Mean: 0.7817425710219672; Std: 0.3823517791015183
hvila-block-layoutlm-finetuned-docbank---
N: 18; Mean: 0.45068281928211157; Std: 0.45471671005951186
hvila-block-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.5749837108144917; Std: 0.45468113195450943
hvila-row-layoutlm-finetuned-docbank---
N: 18; Mean: 0.6385603164993433; Std: 0.38432007142533864
hvila-row-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.8075431628380263; Std: 0.34291789113019977
ivila-block-layoutlm-finetuned-docbank---
N: 18; Mean: 0.6136409163668325; Std: 0.42674025123269516
ivila-block-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.6581086104444326; Std: 0.38131467540590275
s2---
N: 18; Mean: 0.8826942058436625; Std: 0.26666074571314874
-------- ABSTRACT --------
grobid---
N: 18; Mean: 0.7798620127409269; Std: 0.40544842165205675
hvila-block-layoutlm-finetuned-docbank---
N: 18; Mean: 0.8061344751466513; Std: 0.2670278505067842
hvila-block-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.7925158589657537; Std: 0.34406431480167976
hvila-row-layoutlm-finetuned-docbank---
N: 18; Mean: 0.7768576674165859; Std: 0.2687481497765869
hvila-row-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.9083742517833443; Std: 0.2305558534465728
ivila-block-layoutlm-finetuned-docbank---
N: 18; Mean: 0.6711062276188489; Std: 0.2930104554418168
ivila-block-layoutlm-finetuned-grotoap2---
N: 18; Mean: 0.725095298664866; Std: 0.32683248262288206
s2---
N: 18; Mean: 0.7867807167597486; Std: 0.35866026124254563
```




These scores are based on 18 of the documents excluding the following SHAs which fail for VILA:

```
396fb2b6ec96ff74e22ddd2484a9728257cccfbf
3ef6e51baee01b4c90c188a964f2298b7c309b07
4277d1ec41d88d595a0d80e4ab4146d8c2db2539
564a73c07436e1bd75e31b54825d2ba8e4fb68b7
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
