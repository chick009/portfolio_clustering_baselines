import pandas as pd 
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
from sklearn.manifold import TSNE

def find_interval_between(start, end, interval):
    # Set default start and end day to the 1st and last day of the month
    start_date = datetime.strptime(start, '%Y-%m-%d').replace(day=1)
    end_date = datetime.strptime(end, '%Y-%m-%d').replace(day=31)
    
    # Mapping of interval strings to relativedelta objects
    interval_mapping = {'1M': relativedelta(months=1),
                        '3M': relativedelta(months=3),
                        '6M': relativedelta(months=6),
                        '1Y': relativedelta(years=1)}
    
    # Initialize start_lst and end_lst with the first interval
    start_lst = [start_date.strftime('%Y-%m-%d')]
    end_lst = [(start_date + interval_mapping[interval] - relativedelta(days=1)).strftime('%Y-%m-%d')]
    
    # Generate the start and end dates for each interval
    while start_date + interval_mapping[interval] <= end_date:
        start_date += interval_mapping[interval]
        start_lst.append(start_date.strftime('%Y-%m-%d'))
        end_lst.append((start_date + interval_mapping[interval] - relativedelta(days=1)).strftime('%Y-%m-%d'))
        
    return start_lst, end_lst

def process_data(df):
    # Check for missing values in each column
    missing_columns = df.columns[df.isnull().any()]

    # Drop columns with missing values
    df = df.drop(columns=missing_columns)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create a new DataFrame with the scaled data
    scaled_df = pd.DataFrame(scaled_data, index = df.index, columns=df.columns)

    # Return the scaled DataFrame
    return scaled_df

def dim_reduce(method="umap", df=pd.DataFrame):
    """
    Perform dimensionality reduction on a stock DataFrame using a specified manifold learning algorithm.

    Args:
        method (str): The manifold learning algorithm to be used. Options: 'pca', 'tsne', 'umap'.
                      Defaults to 'umap' if not specified.
        df (pd.DataFrame): The stock DataFrame containing the data to be reduced.

    Returns:
        pd.DataFrame: The reduced-dimensional embedding of the stock DataFrame.

    """

    
    embedding = ''

    # If method is not specified, return the original DataFrame
    if method == 'None':
        return df

    # Perform dimensionality reduction using PCA
    if method == 'pca':
        reducer = PCA(n_components=0.95)
        embedding = reducer.fit_transform(df.transpose())

    # Perform dimensionality reduction using t-SNE
    if method == 'tsne':
        reducer = TSNE(n_components=4, method='exact', n_iter=5000)
        embedding = reducer.fit_transform(df.transpose())

    # Perform dimensionality reduction using UMAP
    if method == 'umap':
        reducer = UMAP(n_components=int(df.shape[0] * 0.3), n_neighbors=10, min_dist=0)
        reducer = reducer.fit(df.transpose())
        embedding = reducer.transform(df.transpose())

    # Convert the embedding array into a DataFrame with column names from the original DataFrame
    embedding = pd.DataFrame(embedding.T, columns=df.columns)

    return embedding