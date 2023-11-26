
import torch
import torch.nn as nn
from models import ClusterNet, TAE

def pretrain_autoencoder(trainloader, args, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")

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

    for epoch in range(50):
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

def train_ClusterNET(trainloader, args, verbose):
    """
    function for the ClusterNet Training
    """
    model = ClusterNet(args)
    model = model.to(args.device)

    # MSE Loss Function
    loss_ae = nn.MSELoss()
    loss_kl = nn.KLDivLoss()
    # Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), weight_decay = 0.0001)
    
    # Make the model in training version
    model.train()
    train_loss = 0
    # Training with the data
    for epoch in range(50):
        all_loss = 0
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            optimizer.zero_grad()
            z, x_reconstr, Q, P = model(inputs)

            loss_mse = loss_ae(inputs.squeeze(1), x_reconstr)
            loss_kl = loss_kl(P, Q)
            total_loss = loss_mse + loss_kl

            total_loss.backward()
            optimizer.step()

            all_loss += loss_mse.item()
        
        print(f"For epoch {epoch}, ClusterNet Loss is {all_loss/ batch_idx + 1}")
       