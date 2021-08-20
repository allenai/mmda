# S2agemaker Template
**/stājˈmākər/**

A pun on S2 and AWS SageMaker.

## What is this repo?

This repository provides a template project for packaging up a model.
It implements the AWS SageMaker specification for model inference servers.

Your model will be runnable in a Docker container, and callable over HTTP.
It will also be suitable for deployment into S2 infrastructure.

## What is SageMaker and what is its "specification"?

AWS SageMaker is a service that allows for development and hosting of
machine learning models in the cloud.

It expects models to be packaged in a Docker image following certain conventions.
Images following these conventions can be trivially deployed to AWS SageMaker
and take full advantage of its ecosystem. These conventions are also compatible with
running the image elsewhere, either locally or remotely.

Note: we do not yet support the SageMaker model training specification.

## Ok, I'm working on a new model -- what do I do?

Clone this project and pick a starting branch for your model. Options include:

* `[main]` -- default, use for any simple python 3.8 CPU based model 
* `[pytorch]` -- TODO
* `[allennlp]` -- TODO

Each of these branches provides an appropriate Dockerfile to house your model.

Your main job is to re-implement the stubs in `model`:

#### `model.instance.Instance`
Defines the shape for one object over which inference can be performed.

#### `model.prediction.Prediction`
Defines the shape of the result of inference over one instance.

#### `model.predictor.Predictor`
Used by the included FastAPI server to perform inference. Initialize your model
in the constructor using the supplied `PredictorConfig` instance, and perform inference
for each `Instance` passed via `predict_batch()`. The default batch size is `1`, but
you should handle as many `Instance`s as are provided.

#### `model.predictor.PredictorConfig`
The set of configuration parameters required to instantiate a predictor and
initialize it for inference. This is an appropriate place to specify any parameter 
or configuration values your model requires that aren't packaged with your
versioned model artifacts. These should be rare beyond the included
`artifacts_dir`.

Values for these config fields can be provided as environment variables, see:
`./docker.env`

Will read in ENV vars, see pydantic's 
[setting's management docs](https://pydantic-docs.helpmanual.io/usage/settings/)
for more details.

---
**Please note**:

`ModelConfig`, `Instance`, and `Prediction` should all be implemented as
[pydantic models](https://pydantic-docs.helpmanual.io/usage/models/).

Pydantic allows:
 * static definition of field types
 * imposition of value constraints
 * sophisticated multi-field validation rules
 * marshalling to/from JSON
 
It is also used by FastAPI to generate API documentation for the included server.
 
---
 
## Run Your Model Locally
 
 1. Make sure any artifacts required by your model are in `./artifacts`.
 2. `cd <projectRoot>`
 3. `make serve`
 
This will build your model image and start a server at `localhost:8080`.
 
 
## Invoking Your Model
 
This project implements a single endpoint, `/invocations`, which accepts
`application/json` POST requests.
 
The general format for the JSON body is to provide one or more instances for inference, e.g.:
 
```json
{
    "instances": [ {}, {}, {}, {} ]
}
```

with a response format of:

```json
{
    "predictions": [ {}, {}, {}, {} ]
}
```

For the example project, try:

```bash
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"instances": [{"field1": "asdf", "field2": 1.2}]}' \
    http://localhost:8080/invocations
```

to yield:

```
{"predictions":[{"output_field":"asdf:2.4"}]}
```

## API Documentation

Thanks to `pydantic` and `FastAPI`, you can also view automatically-generated
documentation for your API.

See:

http://locahost:8080/redoc

and

http://localhost:8080/docs

for two different views.

## Tests and CI

This template project provides a CI script to verify the formatting, type safety, and
behavior of your application.

To run:

```bash
cd <projectRoot>
./verify.sh
```

This will build your image, run `mypy`, `pytest` unit tests, and `black` formatting.

These can be run individually via:

`make mypy`

`make unit`

and

`make format`

Add your project's unit tests under `./tests` to include them in the suite.

### Integration Tests

[TODO] single driver script, mechanism for supplying different artifacts

A live server running your model can be run for executing integration tests.

In on terminal:
```
make serve
```

And in another:
```
make it
```

### Loading and Publishing Model Artifacts

[TODO]


