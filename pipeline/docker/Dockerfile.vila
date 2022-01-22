FROM python:3.8-slim-buster

RUN apt-get update && apt-get install nginx iproute2 python3-opencv -y
WORKDIR /src

# Install dependencies
COPY setup.py .
RUN pip install -e .[api]
RUN pip install -e .[vila_predictors]

ARG MODEL_CONFIG
COPY $MODEL_CONFIG /pipeline/vila-model

COPY mmda mmda

ENV PORT 80

CMD ["python", "mmda/predictors/hf_predictors/vila_api.py"]

