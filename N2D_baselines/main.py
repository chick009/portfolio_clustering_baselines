import pandas as pd 
import numpy as np
from utils import find_interval_between, process_data, dim_reduce
from auto_encoder import train_autoencoder
from clustering import choose_hps, cluster_approaches
from metrics_calculation import calculate_cumulative_return, calculate_optimal_portfolio
def prepare_dataset(result_df, data):
    embeddings = ['None', 'pca', 'tsne', 'umap']
    cluster_approaches_lst = ['K-Means', 'Gaussian mixture model', 'Agglomerative Clustering']
    
    for interval in ['1M']:
        start, end = data.index[0], data.index[-1]
        start_lst, end_lst = find_interval_between(start, end, interval)


        for idx in range(len(start_lst)):

            if idx == len(start_lst) - 1:
                continue
            # Locate the time start and time end
            df = data.loc[start_lst[idx]:end_lst[idx]]
            
            # Delete Columns with Missing Values + Scale with min_max_scaler
            df = process_data(df)
      
            for learn_ae in ["yes", "no"]:
                # Learn the time series representation (20 x 500) -> (5 x 500) : Latent Representation
                if learn_ae == "yes":
                    df = train_autoencoder(df)

                for embed in embeddings:
                    
                    df = dim_reduce(embed, df)
                
                    for cluster in cluster_approaches_lst:
                        # print(interval, end_lst[idx], learn_ae, embed, cluster)
                        # Silhouette Coefficient
                        nb_clusters = choose_hps(df, cluster)
                        labels = cluster_approaches(df, nb_clusters, cluster)
                        
                        test_df = data.loc[start_lst[idx + 1]: end_lst[idx + 1]][labels]
                        curr_df = data.loc[start_lst[idx]: end_lst[idx]][labels]
                        
                        metrics_ew = calculate_cumulative_return(test_df, weights = None)
                        metrics_curr = calculate_cumulative_return(curr_df, weights = None)
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Cumulative Return"] = metrics_ew['Cumulative Return']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Sharpe Ratio"] = metrics_ew['Sharpe Ratio']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Sortino Ratio"] = metrics_ew['Sortino Ratio']

                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Current Cumulative Return"] = metrics_curr['Cumulative Return']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Current Sharpe Ratio"] = metrics_curr['Sharpe Ratio']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'EW'), "Current Sortino Ratio"] = metrics_curr['Sortino Ratio']

                        weights = calculate_optimal_portfolio(curr_df)
                        metrics_mvo = calculate_cumulative_return(test_df, weights)

                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Cumulative Return"] = metrics_mvo['Cumulative Return']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Sharpe Ratio"] = metrics_mvo['Sharpe Ratio']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Sortino Ratio"] = metrics_mvo['Sortino Ratio']

                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Current Cumulative Return"] = metrics_curr['Cumulative Return']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Current Sharpe Ratio"] = metrics_curr['Sharpe Ratio']
                        result_df.loc[(interval, end_lst[idx], learn_ae, embed, cluster, 'MVO'), "Current Sortino Ratio"] = metrics_curr['Sortino Ratio']
        
    
data = pd.read_csv('hangseng.csv', index_col=0)
# Assuming your DataFrame is called 'df'
data.index = pd.to_datetime(data.index, format='%m/%d/%Y').strftime('%Y-%m-%d')

interval_lst = ['1M', '3M', '6M', '1Y']
embeddings = ['None', 'pca', 'tsne', 'umap']
cluster_approaches_lst = ['KMeans', 'Gaussian mixture model', 'Agglomerative Clustering']
learn_aes = ["yes", "no"]
multi_index = pd.MultiIndex.from_product([interval_lst, [None], learn_aes, embeddings, cluster_approaches_lst, ['EW', 'MVO']],
                                         names=['interval', 'end', 'learn_ae', 'embeddings', 'cluster', 'weights'])

# Initialize the DataFrame with the multi-index
result_df = pd.DataFrame(index=multi_index)

prepare_dataset(result_df, data)

# result_df.to_csv(...)