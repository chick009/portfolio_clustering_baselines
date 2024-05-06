# Portfolio Allocation Baselines

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)


## Table of Contents

- [Background](#background)
- [Installation](#install)
- [Dataset](#dataset)
- [File Description](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

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

