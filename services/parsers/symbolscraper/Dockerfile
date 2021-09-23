FROM python:3.8-buster

# system deps
RUN apt-get update &&\
        apt-get install -y default-jdk maven poppler-utils nginx &&\
        curl -sSL 'https://github.com/zanibbi/SymbolScraper/archive/bd3b04de61c7cc390d4219358ca0cd95e43aae50.zip' -o symbolscraper.zip &&\
        unzip symbolscraper.zip -d /opt &&\
        mv /opt/SymbolScraper-bd3b04de61c7cc390d4219358ca0cd95e43aae50 /opt/symbolscraper &&\
        rm symbolscraper.zip &&\
        cd /opt/symbolscraper &&\
        make &&\
        ln -sf /opt/symbolscraper/bin/sscraper /bin/sscraper

# mmda deps
WORKDIR /opt/ml/lib
COPY setup.py .
RUN pip install -e .

# server deps
WORKDIR /opt/ml/code
COPY services/parsers/symbolscraper/setup.py .
RUN pip install -e .[dev]

# mmda code
WORKDIR /opt/ml/lib
COPY mmda mmda

# server code
WORKDIR /opt/ml/code
COPY services/parsers/symbolscraper .

# environment
ENV PYTHONPATH "/opt/ml/lib${PYTHONPATH:+:${PYTHONPATH}}"
ENV ARTIFACTS_DIR /opt/ml/model
ENV BATCH_SIZE 1
ENV MODEL_SERVER_TIMEOUT 1200
ENV NUM_WORKERS 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]
