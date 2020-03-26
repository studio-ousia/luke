FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    iputils-ping \
    git \
    python2.7 \
    python-pip \
    python-dev \
    python-software-properties \
    python-tk \
    software-properties-common \
    build-essential \
    cmake \
    libhdf5-dev \
    swig \
    wget \
    curl

## Python 3.6
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y  && \
    apt-get install python3.6 -y \
        python3.6-venv \
        python3.6-dev \
        python3-pip \
        python3-software-properties

# Set Python3.6 as the default python3 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip3 install -U pip

WORKDIR /workspace

RUN pip install Cython
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN git checkout 5633f6dbf7952026264e3aba42413f06752b0515
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /workspace
