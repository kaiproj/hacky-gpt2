FROM python:3.10

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY bin/ bin

ENV PATH="/app/bin:${PATH}"

ARG NO_INIT
ENV NO_INIT=$NO_INIT
RUN init

#ARG NB_USER="jovyan"
#ARG NB_UID="1000"
#ARG NB_GID="100"
#RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}"
#ENV HOME=/home/${NB_USER}

#USER ${NB_USER}

#RUN python build.py
#
## uvicorn --host 0.0.0.0 app:app
#ENTRYPOINT ["uvicorn"]
#
#CMD ["--host", "0.0.0.0", "app:app"]
