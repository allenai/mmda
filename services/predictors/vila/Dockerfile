FROM python:3.8-slim-buster

RUN apt-get update && apt-get install nginx iproute2 python3-opencv -y
WORKDIR /opt/ml/code

# Install dependencies
COPY services/predictors/vila/PROJECT_NAME.txt .
COPY services/predictors/vila/requirements.txt .
COPY services/predictors/vila/setup.py .

RUN pip install -r requirements.txt
RUN pip install --upgrade .[dev]

# Model weights
COPY services/predictors/vila/model-weights /opt/ml/model

# Service code
COPY services/predictors/vila .
# mmda library
COPY mmda mmda

ENV ARTIFACTS_DIR /opt/ml/model
ENV BATCH_SIZE 1
ENV MODEL_SERVER_TIMEOUT 600
ENV NUM_WORKERS 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]

