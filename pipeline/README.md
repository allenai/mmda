```shell script
# install python deps
pip install -e '.[dev,pipeline]'

# Copy VILA model weights into pipeline directory
aws s3 cp --recursive \
  s3://ai2-s2-science-parse-plus-prod/models/ivila/row/s2-vl/ \
  pipeline/vila-model-weights

# start models
docker-compose -f pipeline/docker/pipeline.yml up -d

# run script
python -m pipeline.run path/to/pdf
```  
