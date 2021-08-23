# MMDA - multimodal document analysis

This is work in progress...

There's multiple python projects in this repository.
- [mmda](/mmda) - Defines base classes that parsers
and predictors will implement. Also defines associated data classes.
- [parsers](/parsers) - parser applications
- [predictors](/predictors) - predictor applications

The applications (parsers and predictors) make use of the mmda library.

## Library

### Developing
```bash
conda create -n mmda python=3.8
pip install -e '.[dev]'
```

## Parsers

### [SymbolScraper](https://github.com/zanibbi/SymbolScraper/tree/bd3b04de61c7cc390d4219358ca0cd95e43aae50)

#### Install

Quoted from their `README`
> From the main directory, issue `make`. This will run the Maven build system, 
> download dependencies, etc., compile source files and generate .jar files 
> in `./target`. Finally, a bash script `bin/sscraper` is generated, so that 
> the program can be easily used in different directories.

#### Developing
```bash
cd parsers/symbolscraper
conda create -n sscraper python=3.8
pip install -r requirements.txt
```

## Walkthrough

#### 1. Creating a Document for the first time

In this example, we use the `SymbolScraperParser`. Each parser implements its own `.parse()`.
```python
import os
from mmda.parsers.symbol_scraper_parser import SymbolScraperParser
from mmda.types.document import Document

ssparser = SymbolScraperParser(sscraper_bin_path='...')
doc: Document = ssparser.parse(infile='...pdf', outdir='...', outfname='...json')
```

Because we provided `outdir` and `outfname`, the document is also serialized for you:
```python
assert os.path.exists(os.path.join(outdir, outfname))
```

#### 2. Loading a serialized Document

Each parser implements its own `.load()`.
```python
doc: Document = ssparser.load(infile=os.path.join(outdir, outfname))
```  

#### 3. Iterating through a Document

The minimum requirement for a `Document` is its `.text` field, which is just a `<str>`.

But the usefulness of this library really is when you have multiple different ways of segmenting the `.text`. For example: 

```python
for page in doc.pages:
    print(f'\n=== PAGE: {page.id} ===\n\n')
    for row in page.rows:
        print(row.text)
```

shows two nice aspects of this library:

* `Document` provides iterables for different segmentations of `text`.  Options include `pages, tokens, rows, sents, blocks`.  Not every Parser will provide every segmentation, though.  For example, `SymbolScraperParser` only provides `pages, tokens, rows`.

* Each one of these segments (precisely, `DocSpan` objects) is aware of (and can access) other segment types. For example, you can call `page.rows` to get all Rows that intersect a particular Page.  Or you can call `sent.tokens` to get all Tokens that intersect a particular Sentence.  Or you can call `sent.block` to get the Block(s) that intersect a particular Sentence.  These indexes are built *dynamically* when the `Document` is created and each time a new `DocSpan` type is loaded.  In the extreme, one can do:

```python
for page in doc.pages:
    for block in page.blocks:
        for sent in block.sents:
            for row in sent.rows:
                for token in sent.tokens:
                    pass
```

#### 4. Loading new DocSpan type

Not all Documents will have all segmentations available at creation time. You may need to load new definitions to an existing `Document`.

It's *strongly* recommended to create the full `Document` using a `Parser.load()` but if you need to build it up step by step using the `DocSpan` class and `Document.load()` method: 

```python
from mmda.types.span import Span
from mmda.types.document import Document, DocSpan, Token, Page, Row, Sent, Block

doc: Document(text='I live in New York. I read the New York Times.')
page_jsons = [{'start': 0, 'end': 46, 'id': 0}]
sent_jsons = [{'start': 0, 'end': 19, 'id': 0}, {'start': 20, 'end': 46, 'id': 1}]

pages = [
    DocSpan.from_span(span=Span.from_json(span_json=page_json), 
                      doc=doc, 
                      span_type=Page)
    for page_json in page_jsons
]
sents = [
    DocSpan.from_span(span=Span.from_json(span_json=sent_json), 
                      doc=doc, 
                      span_type=Sent)
    for sent_json in sent_jsons
]

doc.load(sents=sents, pages=pages)

assert doc.sents
assert doc.pages
```

#### 5. Changing the Document

We currently don't support any nice tools for mutating the data in a `Document` once it's been created, aside from loading new data.  Do at your own risk. 

But a note -- If you're editing something (e.g. replacing some `DocSpan` in `tokens`), always call:

```python
Document._build_span_type_to_spans()
Document._build_span_type_to_index()
```  

to keep the indices up-to-date with your modified `DocSpan`.

