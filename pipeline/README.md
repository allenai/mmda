```shell script
# install python deps
pip install -e '.[pipeline]'

# Copy VILA model weights into pipeline directory
aws s3 cp --recursive \
  s3://ai2-s2-science-parse-plus-prod/models/ivila/row/s2-vl/ \
  pipeline/vila-model

# start models
docker-compose -f pipeline/pipeline.yml -d

# run script
python pipeline/run.py path/to/pdf
```  
