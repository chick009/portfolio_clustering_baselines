import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from auto_encoder import AutoEncoder
from tmp_auto_encoder import TmpAutoEncoder
from N2D import N2D
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_autoencoder(train_loader, model = None, model_name = 'auto_encoder', optimizer_choice = 'ADAM', num_epochs = 100, patience = 5):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    first_batch = next(iter(train_loader))
    data = first_batch[0].to(device)
    batch, seq_len, dim = data.shape

    # For Early Stopping Purposes:
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    if model_name == 'auto_encoder' and model == None:
        model = AutoEncoder(seq_len * dim).to(device)
    
    if model_name == 'tmp_auto_encoder' and model == None:
        model = TmpAutoEncoder(dim).to(device)

    # Define your loss function
    loss_fn = nn.MSELoss()

    # Define your optimizer
    if optimizer_choice == 'ADAM':
        optimizer = optim.Adam(model.parameters(), weight_decay = 0.0001, amsgrad = True)

    if optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), 0.001, momentum = 0.9)
        T_0 = 10  # Number of epochs before the first restart
        T_mult = 1  # Multiplicative factor for each restart
        scheduler = CosineAnnealingLR(optimizer, 100)
    
    early_stop = False
    # Train your model
    for epoch in range(num_epochs):
        for batch in train_loader:
            
            inputs = batch[0].to(torch.float32).to(device)  # Get batch data
            
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            if model_name == 'auto_encoder':
        
                inputs = inputs.view(inputs.size(0), -1)

            hidden_repr, outputs = model(inputs)
            # Compute the loss
            loss = loss_fn(outputs, inputs)  # Assuming input data is used as target for reconstruction
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()

            # Learning Rate Schedule also step
            if optimizer_choice == 'SGD':
                scheduler.step()

        # Break the loop if early stopping
        if loss < best_val_loss:
            best_val_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            
            epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                model.load_state_dict(best_state)
                print(f'early stopping occurred at epoch {epoch}')
                early_stop = True
                break

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    return model 

