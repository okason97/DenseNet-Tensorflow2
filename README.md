# DenseNet implementation using Tensorflow 2

## Instalation
```sh
$ pip install densenet
```

## Usage
```python
from densenet import densenet_model
model = densenet_model(classes=n_clases)
```

you can disable the se layers by setting the argument with_se_layers to false.
Changing nb_layers lenght will change the number of dense layers.
