ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter

ADD . /develop
COPY notebooks /tf/notebooks

# Needed for string testing 
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q
RUN apt-get install -y git nano

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install -e /tf/lib/handshape_datasets

RUN pip3 install sklearn
RUN if [[ DOCKER_ENV == "gpu" ]]; then pip3 install tf-nightly-gpu-2.0-preview; fi

RUN mkdir -p /.handshape_datasets
RUN chmod -R a+rwx /.handshape_datasets

WORKDIR /tf
