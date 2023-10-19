import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from N2D import N2D

def multi_label_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # Small value to avoid division by zero
    
    # Clip the predictions to a small range for numerical stability
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Calculate the cross-entropy loss for each label
    loss_per_label = - (targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Average the loss across all labels
    loss = np.mean(loss_per_label)
    
    return loss

def evaluate(batch_data, model, model_name, total_labels):
    # Get the sampling data 

    # K-Means Clustering (Labels) 
    kmean_labels = total_labels[0]
    gmm_labels = total_labels[1]
    agglo_labels = total_labels[2]
    # Gaussian Mixture Model (Labels)
    # Agglomerative Clustering (Labels)

    embedding = []
    model = N2D(model, 8)
    
    for y in batch_data:
      data = y[0].to(torch.float32)

      if model_name == 'auto_encoder':
        data = data.view(data.size(0), -1)
      hidden_repr, _ = model.encoder(data.to('cpu'))
      
      embedding.append(hidden_repr.detach().numpy())

  
    embedding = np.concatenate(embedding)
    if model_name == 'tmp_auto_encoder':
      embedding = embedding[:, -1, :]
    manifold = model.manifold(embedding)
    pred = model.cluster(manifold).argmax(1)

    # Evaluating the final prediction similarity along with 
    print(f'KMeans CE Loss: {multi_label_cross_entropy_loss(pred, kmean_labels)}')
    print(f'GMM CE Loss: {multi_label_cross_entropy_loss(pred, gmm_labels)}')
    print(f'Agglo CE Loss: {multi_label_cross_entropy_loss(pred, agglo_labels)}')

    
    return pred 

def select_stocks(df, cluster_labels):
    selected_stocks = []
    transposed_df = df.transpose()
    # Iterate over each cluster
    print(transposed_df.shape)
    for cluster in range(8):
        # Get the indices of stocks belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # Calculate the centroid of the current cluster
        cluster_centroid = np.mean(transposed_df.iloc[cluster_indices], axis=0)

        # Calculate the Euclidean distance between each stock in the cluster and the centroid
        distances = cdist(transposed_df.iloc[cluster_indices], [cluster_centroid], metric='euclidean')

        # Find the index of the stock with the minimum distance to the centroid
        representative_stock_index = cluster_indices[np.argmin(distances)]

        # Add the representative stock to the selected stocks list
        selected_stocks.append(transposed_df.index[representative_stock_index])

    return selected_stocks
    
def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def compute_metrics(returns):
    # Store metrics in a dictionary 
    metrics = dict()
    
    # Add computation of sharpe ratio and sortino ratio
    metrics["Cumulative Return"] = (returns + 1).cumprod().iloc[-1]
    metrics["Sharpe Ratio"] = np.mean(returns) * np.sqrt(252) / np.std(returns)
    metrics["Sortino Ratio"] = np.mean(returns) * np.sqrt(252) / np.std(returns > 0)
    
    return metrics

def calculate_cumulative_return(df, weights = None):

    # Calculate daily returns
    daily_returns = df.pct_change() + 1
    daily_returns = daily_returns.dropna(how = 'all')

    # Assume equal weighting for all stocks
    if weights is None:
        weights = np.repeat(1/df.shape[1], df.shape[1])
    
    # Calculate portfolio return
    portfolio_return = (weights * daily_returns).sum(axis=1)
    
    # Calculate cumulative return
    cumulative_return = portfolio_return.cumprod()
    
    # Calculate the metrics from the returns
    metrics = compute_metrics(portfolio_return - 1)
    
    return metrics

def calculate_expected_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = calculate_expected_return(weights, returns)
    portfolio_volatility = np.sqrt(calculate_portfolio_variance(weights, cov_matrix))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(weights, returns, cov_matrix, risk_free_rate):
    num_assets = len(returns.columns)
    args = (returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, weights, args=args, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculate_optimal_portfolio(df, cov_matrix = None):
    returns = df.pct_change().dropna(how='all')
    if cov_matrix is None:
        cov_matrix = returns.cov()
    num_assets = len(returns.columns)
    risk_free_rate = 0.01 # Assuming a risk-free rate of 1%
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    result = optimize_portfolio(weights, returns, cov_matrix, risk_free_rate)
    return result.x