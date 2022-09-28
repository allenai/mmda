# MMDA - multimodal document analysis

This is work in progress... Click here for [project status](https://github.com/allenai/mmda/projects/1).

## Setup

```bash
conda create -n mmda python=3.8
pip install -r requirements.txt
```
or
```bash
conda create -n mmda python=3.8
pip install -e '.[dev,<extras_require section from setup.py>]'
```

## Parsers

* [SymbolScraper](https://github.com/zanibbi/SymbolScraper/commit/bd3b04de61c7cc390d4219358ca0cd95e43aae50) - Apache 2.0

    * Quoted from their `README`: From the main directory, issue `make`. This will run the Maven build system, download dependencies, etc., compile source files and generate .jar files in `./target`. Finally, a bash script `bin/sscraper` is generated, so that the program can be easily used in different directories.

* [PDFPlumber](https://github.com/jsvine/pdfplumber) - MIT License    

* [Grobid](https://github.com/kermitt2/grobid) - Apache 2.0    


## Rasterizers

* [PDF2Image](https://github.com/Belval/pdf2image) - MIT License


## Library walkthrough


#### 1. Creating a Document for the first time

In this example, we use the `SymbolScraperParser` to convert a PDF into a bunch of text and `PDF2ImageRasterizer` to convert that same PDF into a bunch of page images.
```python
from typing import List
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer 
from mmda.types.document import Document
from mmda.types.image import PILImage

# PDF to text
ssparser = SymbolScraperParser(sscraper_bin_path='...')
doc: Document = ssparser.parse(input_pdf_path='...pdf')

# PDF to images
pdf2img_rasterizer = PDF2ImageRasterizer()
images: List[PILImage] = pdf2img_rasterizer.rasterize(input_pdf_path='...pdf', dpi=72)

# attach those images to the document
doc.annotate_images(images=images)
```

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


#### 4. Iterating through a Document

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

* Each one of these segments (in our library, we call them `SpanGroup` objects) is aware of (and can access) other segment types. For example, you can call `page.rows` to get all Rows that intersect a particular Page.  Or you can call `sent.tokens` to get all Tokens that intersect a particular Sentence.  Or you can call `sent.rows` to get the Row(s) that intersect a particular Sentence.  These indexes are built *dynamically* when the `Document` is created and each time a new `DocSpan` type is loaded.  In the extreme, one can do:

```python
for page in doc.pages:
    for paragraph in page.paragraphs:
        for sent in paragraph.sents:
            for row in sent.rows:
                for token in sent.tokens:
                    pass
```

You can check which fields are available in a Document via:

```python
doc.fields
> ['pages', 'tokens', 'rows']
```


#### 5. Loading new SpanGroup field

Not all Documents will have all segmentations available at creation time. You may need to load new fields to an existing `Document`.
 
TBD...

#### 6. Editing existing fields in the Document

We currently don't support any nice tools for mutating the data in a `Document` once it's been created, aside from loading new data.  Do at your own risk. 

TBD...



