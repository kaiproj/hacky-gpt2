FROM python:3.10

RUN mkdir /app /built /trained
WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY build.py .
RUN python build.py

COPY sample-data/ sample-data/

COPY *.py .
