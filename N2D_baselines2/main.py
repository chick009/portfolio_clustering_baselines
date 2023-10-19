import argparse
import pandas as pd
import numpy as np

from data_prep import data_prep, process_data
from train import train_autoencoder
from evals import evaluate, select_stocks, calculate_cumulative_return, calculate_optimal_portfolio
from normal_clustering import cluster_approaches

data_combination = [ # argparse
    ('2013-01-01', '2020-01-01', '2020-05-31'), ('2013-01-01', '2021-01-01', '2021-05-31'), ('2013-01-01', '2022-01-01', '2022-05-31')
]

compare_df = pd.read_csv('DJones.csv', index_col=0) # argparse
df = pd.read_csv('DJones_setA.csv', index_col = [0, 1])  # argparse

for start, end, compare in data_combination:

    total_labels = []
    
    for cluster in ['K-Means', 'Gaussian mixture model', 'Agglomerative Clustering']:
        compare_df = process_data(compare_df)
        labels = cluster_approaches(compare_df, 8, cluster)
        total_labels.append(labels)

    model_name = 'tmp_auto_encoder'
    batch_data, train_loader, stock_list = data_prep(df, start, end)
    model = train_autoencoder(train_loader, model_name, 3)
    pred = evaluate(train_loader, model, model_name, total_labels)

    # Select the stocks according to their original time series
    test_df = compare_df.loc[end:compare]
    selected_stock_dc = select_stocks(test_df, pred)
    selected_stock_kmean = select_stocks(test_df, total_labels[0])
    selected_stock_gmm = select_stocks(test_df, total_labels[1])
    selected_stock_agglo = select_stocks(test_df, total_labels[2])

    # Calculate the metrics
    metrics_ew = calculate_cumulative_return(compare_df[selected_stock_dc].loc[end:compare], weights = None)
    metrics_kmean = calculate_cumulative_return(compare_df[selected_stock_kmean].loc[end:compare], weights = None)
    metrics_gmm = calculate_cumulative_return(compare_df[selected_stock_gmm].loc[end:compare], weights = None)
    metrics_agglo = calculate_cumulative_return(compare_df[selected_stock_agglo].loc[end:compare], weights = None)

    print('deep clustering metrics with EW', metrics_ew)
    print('K-means clustering metrics with EW', metrics_kmean)
    print('GMM clustering metrics with EW', metrics_gmm)
    print('Agglo clustering metrics with EW', metrics_agglo)