# The official implementation of Strick

![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-green.svg?style=plastic)

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
  - The cache data and corresponding generation files.
- checkpoints folder
  - The trained model is saved here.
- datasets folder
  - The datasets classes files. If you want to test on your own datasets, try add the corresponding files here.
- modules folder
  - The network structure and loss function files.
- cluster.py
  - Fast Incremental Clustering algorithm.
- eval.py
  - evaluation module.
- self_train.py
  - network self-training module.
- self_train_test.py
  - algorithm testing module.
- test_cluster.py
  - for for testing clustering algorithms.


## Cache Data Preparation
Before training and testing, run the .py file in the "./cache_data" to generation cache data. 

## Training
After setting the configuration, to start self-training, simply run

> python self_train.py

## Testing
Once the training is completed, there will be a saved model in the "checkpoints" specified in the configuration file. 
To test the self-trained model, run

> python self_train_test.py

Or after setting the corresponding configuration file, you can run

> python test_cluster.py

## Dataset
The data includes the road network information and snapshot-related information. Please download from our [repository](https://drive.google.com/drive/folders/1YEwxgkDH0sWR2yMpM2jeb7C2DpknUqqU?usp=sharing).


<!-- ## Citation
If you find this repository useful in your research, please consider citing the following paper:
```

``` -->
