FROM python:3.8-slim-buster

WORKDIR /src

# Install dependencies
RUN apt-get update && apt-get install poppler-utils -y
COPY requirements.txt .
COPY setup.py .
RUN pip install -e .[pipeline]

COPY mmda /src/mmda

ENTRYPOINT ["python"]