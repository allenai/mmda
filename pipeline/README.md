Note: current working directory must be repo root.

```bash
# install python deps
pip install -r requirements.txt

# Copy VILA model weights into pipeline directory
aws s3 cp --recursive \
  s3://ai2-s2-science-parse-plus-prod/models/ivila/row/s2-vl/ \
  pipeline/vila-model

# start models
docker-compose -f pipeline/pipeline.yml up -d

# run script
python pipeline/run.py path/to/pdf
```  
