import argparse
import pandas as pd
import numpy as np
import warnings

# warnings.filterwarnings("ignore")

from data_prep import data_prep, process_data
from train import train_autoencoder
from evals import evaluate, select_stocks, calculate_cumulative_return, calculate_optimal_portfolio
from normal_clustering import cluster_approaches

def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('-bs', type = int, default = 5)
    args.add_argument('-epoch', type = int, default = 1)
    args.add_argument('-k', type = int, default = 5, help = 'num of cluster')
    args.add_argument('-eval_path', type = str, default = 'DJones.csv')
    args.add_argument('-input_path', type = str, default = 'DJones_setA.csv')
    args.add_argument('-model', type = str, default = 'tmp_auto_encoder', help = 'auto_encoder or tmp_auto_encoder')
    args = args.parse_args()

    return args

def main():
    args = get_args()

    data_combination = [
        ('2013-01-01', '2020-01-01', '2020-02-29', '3M'),
        ('2013-01-01', '2020-01-01', '2020-05-31', '6M'), 
        ('2013-01-01', '2021-01-01', '2021-02-28', '3M'),
        ('2013-01-01', '2021-01-01', '2021-05-31', '6M'), 
        ('2013-01-01', '2022-01-01', '2022-02-28', '3M'),
        ('2013-01-01', '2022-01-01', '2022-05-31', '6M')
    ]
    result_df = pd.DataFrame(data_combination, columns=['Start Date', 'End Date', 'Date', 'Period'])

    # Set the multi-index
    result_df.set_index(['Start Date', 'End Date', 'Date', 'Period'], inplace=True)
    print(result_df)

    compare_df = pd.read_csv(args.eval_path, index_col = 0) # argparse
    df = pd.read_csv(args.input_path, index_col = [0, 1])  # argparse
    

    for start, end, compare, interval in data_combination:

        total_labels = []
        model_name = args.model
        batch_data, close_price_df, train_loader, stock_list = data_prep(df, start, end, args.bs)

        for cluster in ['K-Means', 'Gaussian mixture model', 'Agglomerative Clustering']:
            updated_df = process_data(close_price_df)
            labels = cluster_approaches(updated_df.loc[start: end], args.k, cluster)

            total_labels.append(labels)

        model = train_autoencoder(train_loader, model_name, args.epoch)
        pred, k_mean_ce, gmm_ce, agglo_ce = evaluate(train_loader, model, model_name, total_labels)
            # Evaluating the final prediction similarity along with 
        print(f'KMeans CE Loss: {k_mean_ce}')
        print(f'GMM CE Loss: {gmm_ce}')
        print(f'Agglo CE Loss: {agglo_ce}')

        result_df.loc[(start, end, compare, interval), "kmean_ce"] = k_mean_ce
        result_df.loc[(start, end, compare, interval), "gmm_ce"] = k_mean_ce
        result_df.loc[(start, end, compare, interval), "agglo_ce"] = k_mean_ce

        # Select the stocks according to their original time series
        test_df = compare_df[stock_list].loc[start: end]
        
        selected_stock_dc = select_stocks(test_df, pred)
        selected_stock_kmean = select_stocks(test_df, total_labels[0])
        selected_stock_gmm = select_stocks(test_df, total_labels[1])
        selected_stock_agglo = select_stocks(test_df, total_labels[2])

        # Calculate the metrics
        metrics_ew = calculate_cumulative_return(compare_df[selected_stock_dc].loc[end:compare], weights = None)
        metrics_kmean = calculate_cumulative_return(compare_df[selected_stock_kmean].loc[end:compare], weights = None)
        metrics_gmm = calculate_cumulative_return(compare_df[selected_stock_gmm].loc[end:compare], weights = None)
        metrics_agglo = calculate_cumulative_return(compare_df[selected_stock_agglo].loc[end:compare], weights = None)

        result_df.loc[(start, end, compare, interval), "dc_sharpe"] = metrics_ew['Sharpe Ratio']
        result_df.loc[(start, end, compare, interval), "kmean_sharpe"] = metrics_kmean['Sharpe Ratio']
        result_df.loc[(start, end, compare, interval), "gmm_sharpe"] = metrics_gmm['Sharpe Ratio']
        result_df.loc[(start, end, compare, interval), "agglo_sharpe"] = metrics_agglo['Sharpe Ratio']
        
        print('deep clustering metrics with EW', metrics_ew)
        print('K-means clustering metrics with EW', metrics_kmean)
        print('GMM clustering metrics with EW', metrics_gmm)
        print('Agglo clustering metrics with EW', metrics_agglo)

    result_df.to_csv('result_df.csv')
main()