FROM ubuntu:18.04

ARG PYTHON_VERSION=3.8

ENV POETRY_VERSION=1.1.3 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    MINERL_HEADLESS=1

RUN apt-get update
RUN apt-get -y install curl xvfb x11-xserver-utils

# Install Java JDK 8
RUN apt-get install -y openjdk-8-jdk

# Install Python 3
RUN apt-get install -y build-essential software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py

# Install Poetry
RUN python -m pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /minerl-wrappers
COPY ../poetry.lock pyproject.toml /minerl-wrappers/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

COPY .. /minerl-wrappers

# Build minerl
RUN python /minerl-wrappers/tests/build_minerl.py
