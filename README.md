# DenseNet implementation using Tensorflow 2

We implemented Densenet using squeeze and excitation layers in tensorflow 2 for our experiments.

For more information about densenet please refer to the [original paper](https://arxiv.org/abs/1608.06993).

## Model Usage

```python
from densenet import densenet_model
model = densenet_model(classes=n_clases)
```

you can disable the se layers by setting the argument `with_se_layers` to false.
Changing `nb_layers` lenght will change the number of dense layers.

## References

Inspired by flyyufelix keras implementation (https://github.com/flyyufelix/DenseNet-Keras).

For more information about densenet please refer to the original paper (https://arxiv.org/abs/1608.06993).

<details><summary>Dependencies and Installation</summary>

* The code has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFlow 2
* The two main dependencies are TensorFlow and Pillow package (Pillow is included in dependencies)
* To install `densenet` lib run `python setup.py install`
</details>

## Repository Structure

The repository organized as follows. 

- **data** directory contains scripts for dataset downloading and used as a default directory for datasets.

- **densenet** is the library containing the model itself.

- **src/datasets** logic for datasets loading and processing. 

- **src/scripts** directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. tests folder contains basic training procedure on small-valued parameters to check general correctness. results folder contains .md file with current configuration and details of conducted experiments.

## Training

Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/densenet --mode train --config <config>
```

`<config> = cifar10 | cifar100`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/densenet --mode eval --config <config>
```

`<config> = cifar10 | cifar100`

#### Results

In the `results/<ds>` directory you can find the following results of training processes on a specific dataset `<ds>`:

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = cifar10 | cifar100
<model> = densenet
```

To run TensorBoard, use the following command 

```sh
$ tensorboard --logdir=./results/<ds>/summaries/
```

# Environment

## Quickstart

```sh
$ ./bin/start [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu | nightly-gpu-py3
```

<details><summary>or setup and use docker on your own</summary>

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/tf-py3-jupyter.Dockerfile -t <name>:latest .
```

and now run the image

```sh
$ docker run --rm -u $(id -u):$(id -g) -p 6006:6006 -p 8888:8888 <name>:latest
```

</details>

Visit that link, hey look your jupyter notebooks are ready to be created.

If you want, you can attach a shell to the running container

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```

And then you can find the entire source code in `/develop`.

```sh
$ cd /develop
```

To run TensorBoard, use the following command (alternatively python -m tensorboard.main)

```sh
$ tensorboard --logdir=/path/to/summaries
```
