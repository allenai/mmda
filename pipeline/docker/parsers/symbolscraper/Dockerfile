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
WORKDIR /src
COPY setup.py .
RUN pip install -e .[api]

COPY mmda mmda

ENV PORT 80

CMD ["python", "mmda/parsers/symbol_scraper_api.py"]
