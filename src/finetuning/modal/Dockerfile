FROM python:3.10

WORKDIR /app


RUN apt-get update && apt-get install -y cmake

RUN pip install --upgrade pip
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
