FROM python:3.8-buster

RUN apt-get update &&\
    apt-get install nginx iproute2 git poppler-utils curl -y

WORKDIR /src
COPY setup.py .
RUN pip install -e .[api]

COPY mmda mmda

ENV PORT 80
CMD ["python", "mmda/parsers/pdfplumber_api.py"]
