# dev image for running verification checks
FROM python:3.8 AS dev

RUN apt-get update && apt-get install nginx iproute2 python3-opencv -y
WORKDIR /opt/ml/code

COPY services/predictors/layoutparser/PROJECT_NAME.txt .
COPY services/predictors/layoutparser/requirements.txt .
COPY services/predictors/layoutparser/setup.py .

RUN pip install -r requirements.txt
RUN pip install --upgrade .[dev]
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2"

COPY services/predictors/layoutparser .
COPY mmda mmda
ENTRYPOINT ["/bin/bash"]


# server image for running the API
FROM dev AS server

ENV ARTIFACTS_DIR /opt/ml/model
ENV BATCH_SIZE 1
ENV MODEL_SERVER_TIMEOUT 60
ENV NUM_WORKERS 1
ENV PYTHONUNBUFFERED 1

# Download the model weights
RUN python3 entrypoint.py init

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]

