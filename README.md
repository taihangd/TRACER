# The official implementation of TRACER

![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-green.svg?style=plastic)

This is the code for the paper:

- Efficient and Accurate Cross Camera Trajectory Recovery.

## Requirements

- python == 3.9.16
- faiss == 1.7.3
- gensim == 4.3.1
- matplotlib == 3.7.2
- networkx == 2.8.8
- node2vec == 0.4.6
- numpy == 1.24.3
- pyproj == 3.6.0
- PyYAML == 6.0.1
- pytorch == 2.0.1
- tqdm == 4.65.0

The environment is set up with CUDA 11.7. These dependencies can be installed using the following commands:

```bash
conda env create -f environment.yaml
```
or
```bash
conda create --name yourEnv python==3.9.16
conda activate yourEnv
pip install -r requirements.txt
```

## Configuration
There are the configuration files in "./config" folder, where one can edit and set both the training and test options.

## File Description

### code
- cache_data folder
  - the cache data and corresponding generation files.
- config folder
  - the configuration file are saved here.
- datasets folder
  - datasets classes files. If you want to test on your own datasets, try add the corresponding files here.
- modules folder
  - network structure and loss function files.
- universal_functions.py
  - sampling, training, feature extraction functions, etc.
- cluster.py
  - ours Clustering algorithm.
- eval.py
  - evaluation module.
- self_train_stream_data_online_update.py
  - network training module for streaming mode.
- self_train_test_stream_data_online_update.py
  - algorithm testing module for streaming mode.
- self_train_time_slice_sampl.py
  - network training module for batch mode.
- self_train_test_time_slice_sampl.py
  - algorithm testing module for batch mode.

## Cache Data Preparation
Before training and testing, run the .py file in the "./cache_data" to generation cache data. 

## Test
After setting the configuration, to test in batch mode, simply run

```bash
python self_train_time_slice_sampl.py
python self_train_test_time_slice_sampl.py
```

After setting the configuration, to test in streaming mode, simply run

```bash
python self_train_stream_data_online_update.py
python self_train_test_stream_data_online_update.py
```

## Dataset
Please download from our [repository](https://terabox.com/s/1BUll52ghFXuseGRaev-ElA). 
