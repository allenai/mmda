# dev image for running verification checks
FROM python:3.8

RUN apt-get update && apt-get install nginx iproute2 python3-opencv -y

WORKDIR /src
COPY setup.py .
RUN pip install -e .[api]
RUN pip install -e .[lp_predictors]
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2"

COPY mmda mmda

ENV PORT 80
ENV MODEL_CONFIG lp://efficientdet/PubLayNet

CMD ["python", "mmda/predictors/lp_api.py"]

