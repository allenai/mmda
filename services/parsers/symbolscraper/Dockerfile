FROM python:3.8

###########################################################
###### MODEL SPECIFIC SYSTEM DEPS AND SETUP GO HERE #######
###########################################################

# have at it
RUN apt-get update &&\
        apt-get install -y default-jdk maven &&\
        curl -sSL 'https://github.com/zanibbi/SymbolScraper/archive/bd3b04de61c7cc390d4219358ca0cd95e43aae50.zip' -o symbolscraper.zip &&\
        unzip symbolscraper.zip -d /opt &&\
        mv /opt/SymbolScraper-bd3b04de61c7cc390d4219358ca0cd95e43aae50 /opt/symbolscraper &&\
        rm symbolscraper.zip &&\
        cd /opt/symbolscraper &&\
        make &&\
        ln -sf /opt/symbolscraper/bin/sscraper /bin/sscraper

###########################################################
#### S2AGEMAKER REQS BELOW THIS POINT -- DO NOT CHANGE ####
###########################################################

RUN apt-get update && apt-get install nginx -y
WORKDIR /opt/ml/code

COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt
RUN pip install --upgrade .[dev]

COPY . .

ENV ARTIFACTS_DIR /opt/ml/model
ENV BATCH_SIZE 1
ENV MODEL_SERVER_TIMEOUT 60
ENV NUM_WORKERS 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]

