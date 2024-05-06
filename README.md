# Portfolio Allocation Baselines

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)


## Table of Contents

- [Background](#background)
- [Installation](#install)
- [Dataset](#dataset)
- [Folder Description](#usage)
- [Experiment](#run)


## Background 
Our Final Year Projects proposed that financial data is in fact a path of time series, where we could extract the latent representation from the financial data, and perform time series tasks such as clustering

## Installation
This project uses pip packages are enough, please be reminded the codes are run on half a year ago, 
so there may be compatibility issues in the installation, where our torch version is 2.0.1, with cuda 
version as 11.8, please check out https://pytorch.org/get-started/locally/ for more information.

```sh
$ pip install numpy pandas scipy matplotlib seaborn scikit-learn torch torchvision torchaudio umap-learn
```

## Dataset
For both Dow Jones Index and S&P 500 Index, we provided three datasets, 
- Normal: Only contains close price
- setA: Contains only OHLCV financial data
- setB: Contains OHLCV financial data and technical indicators like RSI, MACD, and SMA

## Folder Description:
- main.py: General flows from data preparation to generate results with config as argparse
- data_prep.py: function for preparing the data

- model.py: Deep Temporal Clustering model
- tmp_auto_encoder.py: temporal auto-encoder models using LSTM
- N2D.py: Not too deep clustering model

- normal_clustering.py: clustering functions 
- train_all.py: train all the selected models 
- evals.py: evaluate the results of the model

## Experiments:

Run the shell code as of below, and we could adjust values in the main value with argparse module:
```sh
$ python main.py
```