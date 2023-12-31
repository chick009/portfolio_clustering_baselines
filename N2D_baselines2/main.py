import argparse
import pandas as pd
import numpy as np
import warnings
import torch
 # Show the heatmaps
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from data_prep import data_prep, process_data
from train_all import pretrain_autoencoder, train_ClusterNET, train_autoencoder
from evals import evaluate, select_stocks, calculate_cumulative_return, calculate_optimal_portfolio
from normal_clustering import cluster_approaches
from datetime import datetime

from N2D import N2D
def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('-bs', type=str, default='10')
    args.add_argument('-epoch', type = int, default = 50)
    args.add_argument('-k', type = int, default = 10, help = 'num of cluster')
    args.add_argument('-n_clusters', type = int, default = 10, help = 'num of cluster')
    args.add_argument('-p', type = int, default = 6, help = 'patience for early stopping')
    args.add_argument('-optim', type = str, default = 'ADAM', help = 'Optimizer Choices')
    args.add_argument('-eval_path', type = str, default = 'DJones.csv')
    args.add_argument('-input_path', type = str, default = 'DJones_setA.csv')
    args.add_argument('-model', type = str, default = 'DTC', help = 'auto_encoder or tmp_auto_encoder')
    args.add_argument('-similarity', type = str, default = 'EUC')
    args.add_argument('-n_hidden', type = int, default = 1, help = 'patience for early stopping')
    args.add_argument("-alpha", type=int, default=1, help="alpha hyperparameter for DTC model")
    args.add_argument("-device", type=str, default='cuda:0', help="alpha hyperparameter for DTC model")
    args = args.parse_args()
    
    return args

def main():
    args = get_args()
    batch_size_lst = list(map(int, args.bs.split(',')))
    

    data_combination = [
        ('2013-01-01', '2019-12-31', '2020-01-01', '2020-02-29', '3M'),
        ('2013-01-01', '2019-12-31', '2020-01-01', '2020-05-31', '6M'), 
        ('2013-01-01', '2019-12-31', '2020-01-01', '2020-12-31', '1Y'),
        ('2013-01-01', '2020-12-31', '2021-01-01', '2021-02-28', '3M'),
        ('2013-01-01', '2020-12-31', '2021-01-01', '2021-05-31', '6M'), 
        ('2013-01-01', '2020-12-31', '2021-01-01', '2021-12-31', '1Y'),
        ('2013-01-01', '2021-12-31', '2022-01-01', '2022-02-28', '3M'),
        ('2013-01-01', '2021-12-31', '2022-01-01', '2022-05-31', '6M'),
        ('2013-01-01', '2021-12-31', '2022-01-01', '2022-12-31', '1Y')
    ]
    result_df = pd.DataFrame(data_combination, columns=['Train Start Date', 'Train End Date', 'Test Start Date', 'Test End Date', 'Period'])

    # Set the multi-index
    result_df.set_index(['Train Start Date', 'Train End Date', 'Test Start Date', 'Test End Date', 'Period'], inplace=True)
    # print(result_df)

    compare_df = pd.read_csv(args.eval_path, index_col = 0) # argparse
    df = pd.read_csv(args.input_path, index_col = [0, 1])  # argparse
    

    for start, end, compare_start, compare_end, interval in data_combination:

        total_labels = []
        model_name = args.model
        batch_data, close_price_df, train_loader, stock_list, data_tensor = data_prep(df, start, end, int(batch_size_lst[0]))
        test_df = compare_df[stock_list].loc[start: end]
        model = None
        # ---------------- Performing Clustering using Classical Approaches -------------- #
        for cluster in ['K-Means', 'Gaussian mixture model', 'Agglomerative Clustering']:
            updated_df = process_data(close_price_df)
            labels = cluster_approaches(updated_df.loc[start: end], args.k, cluster)

            total_labels.append(labels)

        
        for batch_sizes in batch_size_lst:
            batch_data, close_price_df, train_loader, stock_list, data_tensor = data_prep(df, start, end, batch_sizes)
            
            # ---------------- Performing Clustering using Deep Temporal Clustering -------------- #
            model_dtc = pretrain_autoencoder(train_loader, args)
            cluster_model = train_ClusterNET(train_loader, model_dtc, data_tensor, args)
            z, x_reconstr, Q, P = cluster_model(data_tensor.to('cuda:0'))
            preds = torch.max(Q, dim=1)[1]
            pred_dtc = preds.cpu().numpy()
            
            del model_dtc
            del cluster_model
            # ---------------- Performing Clustering using N2D Clustering -------------- #
            model = train_autoencoder(train_loader, model, num_epochs = args.epoch, optimizer_choice = args.optim).to("cuda:0")
            model = N2D(model, args.k)
            hidden_repr, _ = model.encoder(data_tensor.to('cuda:0'))
            hidden_repr = hidden_repr.to('cpu').detach().numpy()
            embedding = hidden_repr[:, -1, :]
            manifold = model.manifold(embedding)
            pred_dc = model.cluster(manifold).argmax(1)
            
            
        # Evaluation:
        # z, x_reconstr, Q, P = cluster_model(inputs)
        # preds = torch.max(Q, dim=1)[1]
        # pred, k_mean_ce, gmm_ce, agglo_ce = evaluate(args, train_loader, model, model_name, total_labels)
        
        # Evaluating the final prediction similarity along with 
        # print(f'KMeans CE Loss: {k_mean_ce}')
        # print(f'GMM CE Loss: {gmm_ce}')
        # print(f'Agglo CE Loss: {agglo_ce}')

        # result_df.loc[(start, end, compare_start, compare_end, interval), "kmean_ce"] = k_mean_ce
        # result_df.loc[(start, end, compare_start, compare_end, interval), "gmm_ce"] = gmm_ce
        # result_df.loc[(start, end, compare_start, compare_end, interval), "agglo_ce"] = agglo_ce

        # Select the stocks according to their original time series
        
        selected_stock_dtc = select_stocks(test_df, pred_dtc)
        selected_stock_dc = select_stocks(test_df, pred_dc)
        selected_stock_kmean = select_stocks(test_df, total_labels[0])
        selected_stock_gmm = select_stocks(test_df, total_labels[1])
        selected_stock_agglo = select_stocks(test_df, total_labels[2])
        
        # orders = ['dc', 'kmean', 'gmm', 'agglo']

        
        #for idx, selected_stocks in enumerate([selected_stock_dc, selected_stock_kmean, selected_stock_gmm, selected_stock_agglo]) :
            # scaler = MinMaxScaler()
            # tmp_df = compare_df
            # tmp_df[selected_df] = scaler.fit_transform(compare_df[selected_df])
        #    heatmap = sns.heatmap(close_price_df[selected_stocks].corr())
            #output_folder = r'C:\Users\johnn\final_year_project\N2D_baselines2\heatmaps'
        #    output_filename = f'{orders[idx]}_{interval}_epoch{args.epoch}_{args.input_path}_clusters{args.k}_{args.model}_{start}_{end}_heatmap.png'
        #    output_filepath = f'{output_folder}/{output_filename}'
        #    heatmap.figure.savefig(output_filepath)
        #    plt.clf()
        

        # Calculate the metrics
        metrics_dc = calculate_cumulative_return(compare_df[selected_stock_dc].loc[compare_start:compare_end], weights = None)
        metrics_dtc = calculate_cumulative_return(compare_df[selected_stock_dtc].loc[compare_start:compare_end], weights = None)
        metrics_kmean = calculate_cumulative_return(compare_df[selected_stock_kmean].loc[compare_start:compare_end], weights = None)
        metrics_gmm = calculate_cumulative_return(compare_df[selected_stock_gmm].loc[compare_start:compare_end], weights = None)
        metrics_agglo = calculate_cumulative_return(compare_df[selected_stock_agglo].loc[compare_start:compare_end], weights = None)

        result_df.loc[(start, end, compare_start, compare_end, interval), "dc_sharpe"] = metrics_dc['Sharpe Ratio']
        result_df.loc[(start, end, compare_start, compare_end, interval), "dtc_sharpe"] = metrics_dtc['Sharpe Ratio']
        result_df.loc[(start, end, compare_start, compare_end, interval), "kmean_sharpe"] = metrics_kmean['Sharpe Ratio']
        result_df.loc[(start, end, compare_start, compare_end, interval), "gmm_sharpe"] = metrics_gmm['Sharpe Ratio']
        result_df.loc[(start, end, compare_start, compare_end, interval), "agglo_sharpe"] = metrics_agglo['Sharpe Ratio']
        
        print('deep clustering metrics with EW', metrics_dtc)
        print('deep temporal clustering metrics with EW', metrics_dc)
        print('K-means clustering metrics with EW', metrics_kmean)
        print('GMM clustering metrics with EW', metrics_gmm)
        print('Agglo clustering metrics with EW', metrics_agglo)

    # Get the current datetime as a string
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_df.to_csv(f'result_df_{args.input_path}_epoch{args.epoch}_{args.k}_{current_datetime}.csv')

main()