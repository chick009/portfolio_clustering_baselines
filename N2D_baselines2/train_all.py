
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from models import ClusterNet, TAE
from tmp_auto_encoder import TmpAutoEncoder

def pretrain_autoencoder(trainloader, args, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")  # Fallback to CPU if CUDA is not available

    first_batch = next(iter(trainloader))
    data = first_batch[0].to(device)
    batch, seq_len, dim = data.shape
    
    ## define TAE architecture
    tae = TAE(input_shape = data.shape)
    tae = tae.to(device)

    ## MSE loss
    loss_ae = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), weight_decay = 0.0001)
    tae.train()

    for epoch in range(10):
        all_loss = 0
        for batch_idx, inputs in enumerate(trainloader):
            inputs = inputs[0].to(torch.float32).to(device)
            # inputs = inputs.type(torch.FloatTensor).to(args.device)
            optimizer.zero_grad()

            # Output Shape (B x H * Pooling)
            z, x_reconstr = tae(inputs)
            loss_mse = loss_ae(inputs, x_reconstr.squeeze(3)) # 

            loss_mse.backward()
            all_loss += loss_mse.item()
            optimizer.step()

        print(f"Pretraining autoencoder loss for epoch {epoch} is : {all_loss / (batch_idx + 1)}")
    
    return tae 

def train_ClusterNET(trainloader, trained_model, data_tensor, args):
    """
    function for the ClusterNet Training
    """
    model = ClusterNet(args, trained_model)
    model = model.to(args.device)
    model.init_centroids(data_tensor)
    # MSE Loss Function
    loss_ae = nn.MSELoss()
    loss_kl = nn.KLDivLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.0001)
    
    # Make the model in training version
    model.train()
    train_loss = 0
    # Training with the data
    for epoch in range(10):
        all_loss = 0
        for batch_idx, inputs in enumerate(trainloader):
            inputs = inputs[0].to(torch.float32).to(args.device)
            optimizer.zero_grad()
            z, x_reconstr, Q, P = model(inputs)

            loss_mse = loss_ae(inputs, x_reconstr.squeeze(3))
            loss_kl2 = loss_kl(P, Q)
            total_loss = loss_mse + loss_kl2 * 0.2

            # 
            total_loss.backward()
            optimizer.step()
            all_loss += loss_mse.item()
        
        print(f"For epoch {epoch}, ClusterNet Loss is {all_loss/ batch_idx + 1}")
    return model


def train_autoencoder(train_loader, model = None, model_name = 'tmp_auto_encoder', optimizer_choice = 'SGD', num_epochs = 100, patience = 10):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    first_batch = next(iter(train_loader))
    data = first_batch[0].to(device)
    batch, seq_len, dim = data.shape

    # For Early Stopping Purposes:
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    model = TmpAutoEncoder(dim, seq_len).to(device)

    # Define your loss function
    loss_fn = nn.MSELoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay = 0.0001, amsgrad = True)

    
    early_stop = False
    # Train your model
    for epoch in range(num_epochs):
        for batch in train_loader:
            
            inputs = batch[0].to(torch.float32).to(device)  # Get batch data
            
            optimizer.zero_grad()  # Zero the gradients

            hidden_repr, outputs = model(inputs)
            # Compute the loss
            loss = loss_fn(outputs, inputs)  # Assuming input data is used as target for reconstruction
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()

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

