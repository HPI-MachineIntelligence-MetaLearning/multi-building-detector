# multi-building-detector
Multibox detection framework with interchangeable architectures

# Installation
You need to install miniconda first. 

## GPU Install  
```
conda env create -f environment-gpu.yml
source activate multi-building-detector-gpu

```

## CPU Install
```
conda env create -f environment.yml
source activate multi-building-detector

```


# Configuration
Save a new `*.yml` file in the config directory.
Currently following fields are supported:

```
input: <data input path>
output: <training result output path>
batch_size: <batch size>
iterator: <chainer iterator class>
device: <gpu index or -1 for cpu>
```

# Usage
```
usage: python -m multibuildingdetector [-h] -c CONFIG

Running configured multibox detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to config

```