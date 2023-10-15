import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def denoise_cov_matrix(df):
    
    # Normalize the covariance matrix
    returns = df.pct_change().dropna(how='all')
    cov_matrix = returns.cov()
    cov_matrix = pd.DataFrame(scaler.fit_transform(cov_matrix))
    
    # Assuming your DataFrame is named 'df'
    kernel_matrix = pairwise_kernels(cov_matrix.values, metric='rbf')

    # Convert the kernel matrix to a DataFrame
    kernel_df = pd.DataFrame(kernel_matrix)
    
    # Spectral Decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(kernel_df.values)
    
    # Setting the theorectically bound for eigenvalues
    tuta = 11 / 11 
    max_lambda = 1 + tuta + 2*tuta
    min_lambda = 1 + tuta - 2*tuta
    
    # Replacing the eigenvalues that is out of the theorectical bound
    eigen_new = np.where(eigenvalues > max_lambda, np.mean(eigenvalues) , eigenvalues)
    eigen_new = np.where(eigen_new < min_lambda, np.mean(eigen_new) , eigen_new)
    
    # Reconstruct the covariance matrix
    diag_matrix = np.diag(eigenvalues)
    reconstructed_matrix = pd.DataFrame(eigenvectors @ diag_matrix @ eigenvectors.T)
    
    return reconstructed_matrix

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

def denoise_cov_matrix(df):
    
    # Normalize the covariance matrix
    scaler = MinMaxScaler()
    returns = df.pct_change().dropna(how='all')
    cov_matrix = returns.cov()
    cov_matrix = pd.DataFrame(scaler.fit_transform(cov_matrix))
    
    # Assuming your DataFrame is named 'df'
    kernel_matrix = pairwise_kernels(cov_matrix.values, metric='rbf')

    # Convert the kernel matrix to a DataFrame
    kernel_df = pd.DataFrame(kernel_matrix)
    
    # Spectral Decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(kernel_df.values)
    
    # Setting the theorectically bound for eigenvalues
    tuta = 11 / 11 
    max_lambda = 1 + tuta + 2*tuta
    min_lambda = 1 + tuta - 2*tuta
    
    # Replacing the eigenvalues that is out of the theorectical bound
    eigen_new = np.where(eigenvalues > max_lambda, np.mean(eigenvalues) , eigenvalues)
    eigen_new = np.where(eigen_new < min_lambda, np.mean(eigen_new) , eigen_new)
    
    # Reconstruct the covariance matrix
    diag_matrix = np.diag(eigenvalues)
    reconstructed_matrix = pd.DataFrame(eigenvectors @ diag_matrix @ eigenvectors.T)
    
    return reconstructed_matrix