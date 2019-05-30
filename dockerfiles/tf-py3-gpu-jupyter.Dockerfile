FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

RUN apt-get update -q
RUN apt-get install -y git
RUN apt-get clean

RUN python3 --version

RUN pip3 install tf-nightly-gpu-2.0-preview
RUN pip3 install -U scikit-learn

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install -e /tf/lib/handshape_datasets

WORKDIR /tf
