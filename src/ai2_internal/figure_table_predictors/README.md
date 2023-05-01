# Example of making a call to FAST API for figure_table_predictor once it is running locally

```python
def process_table_figure(doc):
    url='http://localhost:8080/invocations'
    tokens = [api.SpanGroup.from_mmda(sg) for sg in doc.tokens]
    rows = [api.SpanGroup.from_mmda(sg) for sg in doc.rows]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
    vila_span_groups = [api.SpanGroup.from_mmda(sg) for sg in doc.vila_span_groups]
    blocks = [api.SpanGroup.from_mmda(sg) for sg in doc.blocks]
    pages = [api.SpanGroup.from_mmda(sg) for sg in doc.pages]
    images = [tobase64(image) for image in doc.images]
    data={
      "instances": [{
      "symbols": doc.symbols,
      "tokens": [token.dict() for token in tokens],
      "rows": [token.dict() for token in rows],
      "pages": [token.dict() for token in pages],
      "vila_span_groups": [token.dict() for token in vila_span_groups],
      "blocks": [token.dict() for token in blocks],
      "images": images}]}
    result = requests.post(url, json=data)
    result.raise_for_status()
    return result
```

# Performance of the model evaluated using annotators are as follows:
![performance_metrics.png](performance_metrics.png)

Exact match column means that the model predicted figure/table and captions exactly as the annotators found it 
reasonable, partial match means that some information was lost.

Here is the link to the [annotation_set](https://docs.google.com/spreadsheets/d/1dVLBhE9G6_rHNMIX0Cv6jqaMgviVOTv7UMA2dISuNo8/edit#gid=379619488&range=T4:Z12)
Link to the design considerations [design_considerations](https://docs.google.com/document/d/1Kfbcyw7pXC4uJCiodhUn_vgnA2J981gUK7KbQtkaG5Y/edit?usp=sharing)
