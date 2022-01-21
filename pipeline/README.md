```shell script
# Copy VILA model weights into pipeline directory
aws s3 cp --recursive s3://ai2-s2-research/scienceparseplus/temporary-experiment-results/s2-hard-visionline-few-shot-cv-v2/hierarchical_model-block/weak-strong-layoutlm-average-first/70/0/ pipeline/vila-model-weights

# start models
docker-compose -f pipeline/docker/pipeline.yml up -d

# run script
python -m pipeline.run path/to/pdf
```  
