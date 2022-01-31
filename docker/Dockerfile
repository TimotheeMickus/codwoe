FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa

# set up python
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    iputils-ping \
    git \
    curl \
    build-essential \
    cmake \
    libhdf5-dev \
    swig \
    wget \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-pip \
    python3-software-properties

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8

# Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# install libraries
RUN pip3 install -U --no-cache-dir \
	absl-py==0.12.0 \
	cachetools==4.2.2 \
	certifi==2020.12.5 \
	chardet==4.0.0 \
	click==8.0.1 \
	cycler==0.10.0 \
	filelock==3.0.12 \
	google-auth==1.30.0 \
	google-auth-oauthlib==0.4.4 \
	grpcio==1.37.1 \
	huggingface-hub==0.0.12 \
	idna==2.10 \
	joblib==1.0.1 \
	kiwisolver==1.3.1 \
	Markdown==3.3.4 \
	matplotlib==3.4.2 \
	moverscore==1.0.3 \
	nltk==3.6.7 \
	numpy>=1.20.3 \
	oauthlib==3.1.0 \
	packaging==21.0 \
	Pillow==8.3.0 \
	portalocker==2.3.0 \
	protobuf==3.17.0 \
	pyasn1==0.4.8 \
	pyasn1-modules==0.2.8 \
  pyemd \
	pyparsing==2.4.7 \
	python-dateutil==2.8.1 \
	PyYAML==5.4.1 \
	regex==2022.1.18 \
	requests==2.25.1 \
	requests-oauthlib==1.3.0 \
	rsa==4.7.2 \
	sacremoses==0.0.45 \
	sentencepiece==0.1.96 \
	six==1.16.0 \
	tensorboard==2.5.0 \
	tensorboard-data-server==0.6.1 \
	tensorboard-plugin-wit==1.8.0 \
	tokenizers==0.8.1rc2 \
	torch==1.8.1 \
	tqdm==4.60.0 \
	transformers==3.1.0 \
	typing==3.7.4.3 \
	typing-extensions==3.10.0.0 \
	urllib3==1.26.4 \
	Werkzeug==2.0.1

# the next line will patch moverscore so that it runs on cpu, rather than on your cuda:0 device.
# comment this line if you have access to a GPU
RUN find . -type f -name moverscore_v2.py -exec sed -i 's/cuda:0/cpu/g' {} \;
RUN find . -type f -name moverscore_v2.py -exec sed -i '2 i\import os' {} \;
RUN find . -type f -name moverscore_v2.py -exec sed -i "s/model_name = 'distilbert-base-uncased'/model_name = os.environ.get('MOVERSCORE_MODEL', 'distilbert-base-uncased')/g" {} \;


RUN python3 -c "import nltk; nltk.download('punkt');"
RUN python3 -c "import os; os.environ['MOVERSCORE_MODEL'] = 'distilbert-base-multilingual-cased' ; import moverscore_v2"
