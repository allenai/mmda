# MMDA - multimodal document analysis

This is work in progress... Click here for [project status](https://github.com/allenai/mmda/projects/1).

## Setup

```bash
conda create -n mmda python=3.8
pip install -e '.[dev,<extras_require section from setup.py>]'
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

## Quickstart guide

#### 1. Create a Document for the first time from a PDF

In this example, we use the `PdfPlumberParser` to convert a PDF into a bunch of text and `PDF2ImageRasterizer` to convert that same PDF into a bunch of page images.
```python
from typing import List
from mmda.parsers import PDFPlumberParser
from mmda.rasterizers import PDF2ImageRasterizer 
from mmda.types import Document, PILImage

# PDF to text
parser = PDFPlumberParser()
doc: Document = parser.parse(input_pdf_path='...pdf')

# PDF to images
rasterizer = PDF2ImageRasterizer()
images: List[PILImage] = rasterizer.rasterize(input_pdf_path='...pdf', dpi=72)

# attach those images to the document
doc.annotate_images(images=images)
```

#### 2. Iterating through a Document

The minimum requirement for a `Document` is its `.symbols` field, which is just a `<str>`.  For example:

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

* `.metadata: Metadata`, A free 

    * **Span-Box Coupling:** Every `Span` is associated with a single `Box`, and not a `BoxGroup`. In this library, we restrict all of our `Span` to be units that can be represented by a single rectangular box. This is instead of allowing *any* (start, end) which would result in spans that can't necessarily be cleanly represented by a single box.
    * 

**FAQS**

Q. Why do we need `BoxGroup` if we already have `Box` in each `Span`?

A: Let's consider a `SpanGroup` object representing a single sentence in a paper. We know a single `Box` can't properly cover a sentence, because sentences can wrap rows & even cross columns/page:

* One way to represent the visual area of that sentence is to take the Union of all `Box` in every involved `Span` -- This leaves us with many rectangles. 
* But another way to synthesize all those `Box` into one giant `Box` (which might even overlap other text outside of this sentence). 
* Finally, a third way is to synthesize all the `Box` of tokens on the same row into one `Box`, but keep `Box` on different rows separate. None of these ways 
    

#### 5. Adding a new SpanGroup field

Not all Documents will have all segmentations available at creation time. You may need to load new fields to an existing `Document`. This is where `Predictor` comes in:

```python
from mmda.predictors.lp_predictors import LayoutParserPredictor

predictor = LayoutParserPredictor(model='lp://efficientdet/PubLayNet')

output = predictor.predict(document=doc)

```
 




## Parsers

* [PDFPlumber](https://github.com/jsvine/pdfplumber) - MIT License    


## Rasterizers

* [PDF2Image](https://github.com/Belval/pdf2image) - MIT License


## Predictors


## Library walkthrough




#### 2. Saving a Document

You can convert a Document into a JSON object.

```python
import os
import json

# usually, you'll probably want to save the text & images separately:
with open('...json', 'w') as f_out:
    json.dump(doc.to_json(with_images=False), f_out, indent=4)

os.makedirs('.../', exist_ok=True)
for i, image in enumerate(doc.images):
    image.save(os.path.join('.../', f'{i}.png'))
    
    
# you can also save images as base64 strings within the JSON object
with open('...json', 'w') as f_out:
    json.dump(doc.to_json(with_images=True), f_out, indent=4)
```


#### 3. Loading a serialized Document

You can create a Document from its saved output.

```python
import json
import os

from mmda.document import Document
from typing import List
from mmda.types.image import PILImage, pilimage

# directly from a JSON.  This should handle also the case where `images` were serialized as base64 strings.
with open('...json') as f_in:
    doc_dict = json.load(f_in)
    doc = Document.from_json(doc_dict=doc_dict)

# if you saved your images separately, then you'll want to reconstruct them & re-attach
images: List[PILImage] = []
for i, page in enumerate(doc.pages):
    image_path = os.path.join(outdir, f'{i}.png')
    assert os.path.exists(image_path), f'Missing file for page {i}'
    image = pilimage.open(image_path)
    images.append(image)
doc.annotate_images(images=images)
```  



#### 6. Editing existing fields in the Document

We currently don't support any nice tools for mutating the data in a `Document` once it's been created, aside from loading new data.  Do at your own risk. 

TBD...



