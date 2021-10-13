# Processing pipeline

To build a pipeline, the different parsers and predictors are exposed as local HTTP services
running within Docker, and stitched together in a Python script.

The pipeline is implemented in the [run.py](run.py) script and can be run using 

```shell script
pipeline/run.sh 
```

The models can be customized by modifying 
[pipeline.yml](docker/pipeline.yml) or, if necessary, the model's corresponding
`Dockerfile`.

## Example

```shell script
# Copy VILA model weights into pipeline directory
aws s3 cp --recursive s3://ai2-s2-research/scienceparseplus/temporary-experiment-results/s2-hard-visionline-few-shot-cv-v2/hierarchical_model-block/weak-strong-layoutlm-average-first/70/0/ pipeline/vila-model-weights

# Copy input PDF into pipeline directory
aws s3 cp s3://ai2-s2-pdfs/33f0/e4823ca258a2a5f44878da883ca2508c044d.pdf pipeline/33f0e4823ca258a2a5f44878da883ca2508c044d.pdf

# Run pipeline
pipeline/run.sh 33f0e4823ca258a2a5f44878da883ca2508c044d.pdf 
```  
