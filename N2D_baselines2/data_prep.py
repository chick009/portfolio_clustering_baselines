import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def data_prep(df, start = '2013-01-01', end = '2020-01-01', batch_size = 8):
    
    # Initialize a 3D array that can be appended in the third dimension
    batch_data = []
    stock_list = []
    close_price_df = pd.DataFrame()
    # Iterate over each stock's data
    for stock in df.columns:
       
        # Pivot the tables for converting shape
        df_pivoted = df[stock].unstack().reset_index()
        df_pivoted.index = df_pivoted['first']
        df_piv = df_pivoted.drop(columns= ['first'])
        df_piv = df_piv.loc[start: end]        

        # 1. Check if the data has missing values -> if yes, then continue
        if df_piv.isnull().any().any():
            continue

        close_price_df[stock] = df_piv['Adj Close']
        
        # 2. Scale the data using min_max_scaler to range(0, 1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(df_piv.values)
      
        # 3. Add the data into a batch
        batch_data.append(scaled_data)
        stock_list.append(stock)
    
    # Convert the data array to a PyTorch tensor
    data_tensor = torch.tensor(batch_data, dtype=torch.float32)

    # Create a TensorDataset without labels
    dataset = TensorDataset(data_tensor)

    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    return np.array(batch_data), close_price_df, train_loader, stock_list


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
