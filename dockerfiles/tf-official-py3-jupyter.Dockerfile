ARG DOCKER_ENV=nightly-gpu-py3

FROM tensorflow/tensorflow:${DOCKER_ENV}-jupyter

ADD . /develop
COPY notebooks /tf/notebooks

RUN apt-get update -q
RUN apt-get install -y git nano

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install -e /tf/lib/handshape_datasets

RUN pip3 install -e /develop/protonet-tf
RUN pip3 install tf-nightly-gpu-2.0-preview
RUN pip3 install sklearn

RUN mkdir -p /.handshape_datasets
RUN chmod -R a+rwx /.handshape_datasets

RUN chmod -R a+rwx /tf
RUN chmod -R a+rwx /develop

WORKDIR /tf
