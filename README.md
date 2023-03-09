# MMDA - multimodal document analysis

This is work in progress... 

## Setup

```bash
conda create -n mmda python=3.8
pip install -e '.[dev,<extras_require section from setup.py>]'
```

For most users, we recommend using recipes:
```python
pip install -e '.[dev,recipes]'
```

## Unit testing
Note that pytest is running coverage, which checks the unit test coverage of the code.
The percent coverage can be found in setup.cfg file.
```bash
pytest
```
for latest failed test
```bash
pytest --lf --no-cov -n0
```
for specific test name of class name
```bash
pytest -k 'TestFigureCaptionPredictor' --no-cov -n0
```

## Quick start


#### 1. Create a Document for the first time from a PDF

In this example, we use the `CoreRecipe` to convert a PDF into a bunch of text and images.
```python
from mmda.types import Document
from mmda.recipes import CoreRecipe

recipe = CoreRecipe()
doc: Document = recipe.from_path(pdfpath='...pdf')
```

#### 2. Understanding the output: the `Document` class

What is a `Document`? At minimum, it is some text, saved under the `.symbols` field, which is just a `<str>`.  For example:

```python
doc.symbols
> "Language Models as Knowledge Bases?\nFabio Petroni1 Tim Rockt..."
```

But the usefulness of this library really is when you have multiple different ways of segmenting `.symbols`. For example, segmenting the paper into Pages, and then each page into Rows: 

```python
for page in doc.pages:
    print(f'\n=== PAGE: {page.id} ===\n\n')
    for row in page.rows:
        print(row.symbols)
        
> ...
> === PAGE: 5 ===
> ['tence x, s′ will be linked to s and o′ to o. In']
> ['practice, this means RE can return the correct so-']
> ['lution o if any relation instance of the right type']
> ['was extracted from x, regardless of whether it has']
> ...
```

shows two nice aspects of this library:

* `Document` provides iterables for different segmentations of `symbols`.  Options include things like `pages, tokens, rows, sents, paragraphs, sections, ...`.  Not every Parser will provide every segmentation, though.  For example, `SymbolScraperParser` only provides `pages, tokens, rows`.  More on how to obtain other segmentations later.

* Each one of these segments (in our library, we call them `SpanGroup` objects) is aware of (and can access) other segment types. For example, you can call `page.rows` to get all Rows that intersect a particular Page.  Or you can call `sent.tokens` to get all Tokens that intersect a particular Sentence.  Or you can call `sent.rows` to get the Row(s) that intersect a particular Sentence.  These indexes are built *dynamically* when the `Document` is created and each time a new `SpanGroup` type is loaded.  In the extreme, one can do:

```python
for page in doc.pages:
    for paragraph in page.paragraphs:
        for sent in paragraph.sents:
            for row in sent.rows: 
                ...
```

as long as those fields are available in the Document. You can check which fields are available in a Document via:

```python
doc.fields
> ['pages', 'tokens', 'rows']
```

#### 3. Understanding intersection of SpanGroups

Note that `SpanGroup` don't necessarily perfectly nest each other. For example, what happens if:

```python
for sent in doc.sents:
    for row in sent.rows:
        print([token.symbols for token in row.tokens])
```

Tokens that are *outside* each sentence can still be printed. This is because when we jump from a sentence to its rows, we are looking for *all* rows that have *any* overlap with the sentence. Rows can extend beyond sentence boundaries, and as such, can contain tokens outside that sentence.

Here's another example:
```python
for page in doc.pages:
    print([sent.symbols for sent in page.sents])
```

Sentences can cross page boundaries. As such, adjacent pages may end up printing the same sentence.

But
```python
for page in doc.pages:
    print([row.symbols for row in page.rows])
    print([token.symbols for token in page.tokens])
``` 
rows and tokens adhere strictly to page boundaries, and thus will not repeat when printed across pages.

A key aspect of using this library is understanding how these different fields are defined & anticipating how they might interact with each other. We try to make decisions that are intuitive, but we do ask users to experiment with fields to build up familiarity.




#### 4. What's in a `SpanGroup`?

Each `SpanGroup` object stores information about its contents and position:

* `.spans: List[Span]`, A `Span` is a pointer into `Document.symbols` (that is, `Span(start=0, end=5)` corresponds to `symbols[0:5]`) and a single `Box` representing its position & rectangular region on the page.

* `.box_group: BoxGroup`, A `BoxGroup` object stores `.boxes: List[Box]`.  

* `.metadata: Metadata`, A free form dictionary-like object to store extra metadata about that `SpanGroup`. These are usually empty. 



#### 5. How can I manually create my own `Document`?

If you look at what is happening in `CoreRecipe`, it's basically stitching together 3 types of tools: `Parsers`, `Rasterizers` and `Predictors`.

* `Parsers` take a PDF as input and return a `Document` compared of `.symbols` and other fields. The example one we use is a wrapper around [PDFPlumber](https://github.com/jsvine/pdfplumber) - MIT License utility.

* `Rasterizers` take a PDF as input and return an `Image` per page that is added to `Document.images`. The example one we use is [PDF2Image](https://github.com/Belval/pdf2image) - MIT License. 

* `Predictors` take a `Document` and apply some operation to compute a new set of `SpanGroup` objects that we can insert into our `Document`. These are all built in-house and can be either simple heuristics or full machine-learning models.


If we look at how `CoreRecipe` is implemented, what's happening in `.from_path()` is:

```
    def from_path(self, pdfpath: str) -> Document:
        logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdfpath)

        logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdfpath, dpi=72)
        doc.annotate_images(images=images)

        logger.info("Predicting words...")
        words = self.word_predictor.predict(document=doc)
        doc.annotate(words=words)

        logger.info("Predicting blocks...")
        blocks = self.effdet_publaynet_predictor.predict(document=doc)
        equations = self.effdet_mfd_predictor.predict(document=doc)
        doc.annotate(blocks=blocks + equations)

        logger.info("Predicting vila...")
        vila_span_groups = self.vila_predictor.predict(document=doc)
        doc.annotate(vila_span_groups=vila_span_groups)

        return doc
```

You can see how the `Document` is first created using the `Parser`, then `Images` are added to the `Document` by using the `Rasterizer` and `.annotate_images()` method. Then we layer on multiple `Predicors` worth of predictions, each added to the `Document` using `.annotate()`.

#### 6. How can I save my `Document`?

```python
import json
with open('filename.json', 'w') as f_out:
    json.dump(doc.to_json(with_images=True), f_out, indent=4)
```

will produce something akin to:
```python
{
    "symbols": "Language Models as Knowledge Bases?\nFabio Petroni1 Tim Rockt...",
    "images": "...",
    "rows": [...],
    "tokens": [...],
    "words": [...],
    "blocks": [...],
    "vila_span_groups": [...]
}
```

Note that `Images` are serialized to `base64` if you include `with_images` flag. Otherwise, it's left out of JSON serialization by default.

#### 7. How can I load my `Document`?

These can be used to reconstruct a `Document` again via:

```python
with open('filename.json') as f_in:
    doc_dict = json.load(f_in)
    doc = Document.from_json(doc_dict)
```

