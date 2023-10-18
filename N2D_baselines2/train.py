import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from auto_encoder import AutoEncoder
from N2D import N2D

def train_autoencoder(train_loader,  model = 'auto_encoder', num_epochs = 100):
    first_batch = next(iter(train_loader))
    data = first_batch[0]
    batch, seq_len, dim = data.shape

    if model == 'auto_encoder':
        model = AutoEncoder(seq_len * dim)
    
    # Define your loss function
    loss_fn = nn.MSELoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters())
    
    # Train your model
    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs = batch[0].to(torch.float32)  # Get batch data
            
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            inputs = inputs.view(inputs.size(0), -1)
          
            hidden_repr, outputs = model(inputs)
            # Compute the loss
            loss = loss_fn(outputs, inputs)  # Assuming input data is used as target for reconstruction
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    return model 

