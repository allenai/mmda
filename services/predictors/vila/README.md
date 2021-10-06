# VILA Model API 

A SageMaker-compatible REST API exposing VILA predictors

## Running The Model Locally
 
`make serve` will build your model image and start a server at `localhost:8080`.
 
 Check http://localhost:8080/docs for API documentation
 
The request body is of the form:
 
 ```json
{
  "instances": [{doc1}, {doc2}, ...]
}
```
where each `doc` object comes from `mmda.types.document.Document.to_json()`

The response is of the form:

```json
{
  "predictions": [[doc1Ann1, doc2Ann2, ...], [doc2Ann1, doc2Ann2, ...], ...]
}
```
where each `docAnn` object can be deserialized with `SpanGroup.from_json()` 
 
 
## Tests and CI

Full build verification:

```bash
./verify.sh
```

Reformat code:
```shell script
make reformat
```

To run integration tests fully within Docker:
```shell script
make serve-background
make integration-tests
make kill-server
```

To debug the service and/or tests locally:
```shell script
make model-weights
ARTIFACTS_DIR=. python server/api.py
pytest integration_tests
```
