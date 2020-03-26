FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y build-essential software-properties-common bash curl git python3 python3-pip

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get install -y python3.6 python3.6-dev python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

WORKDIR /workspace

RUN pip3 install Cython
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN git checkout 5633f6dbf7952026264e3aba42413f06752b0515
RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN transformers-cli download --cache-dir=/workspace/.cache/transformers roberta-large
RUN transformers-cli download --cache-dir=/workspace/.cache/transformers roberta-base

WORKDIR /workspace
