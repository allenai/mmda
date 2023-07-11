# parsers


### example usage - grobid parser

1. Get Grobid running locally.  For example:


```bash
docker run -t --rm --init -p 8070:8070 lfoppiano/grobid:0.7.0
```

will use a pre-built Docker image.

2. In Python:

```python
from mmda.types.document import Document
from mmda.parsers.grobid_parser import GrobidHeaderParser

grobid_parser = GrobidHeaderParser(url='http://localhost:8070/api/processHeaderDocument')
doc: Document = grobid_parser.parse(input_pdf_path='...pdf', output_json_path='...json', tempdir='.../')

doc.title
doc.abstract
```
