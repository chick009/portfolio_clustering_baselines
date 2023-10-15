import torch
import pandas as pd
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, flatten_dim):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(Encoder(flatten_dim, int(flatten_dim // 1.25), True),
                                     Encoder(int(flatten_dim // 1.25), int(flatten_dim // 1.5), True),
                                     Encoder(int(flatten_dim // 1.5), int(flatten_dim // 1.7), True))
        
        self.decoder = nn.Sequential(Decoder(int(flatten_dim // 1.7), int(flatten_dim // 1.5), True),
                                     Decoder(int(flatten_dim // 1.5), int(flatten_dim // 1.25), True),
                                     Decoder(int(flatten_dim // 1.25), int(flatten_dim), True))
            
    def forward(self, x):
        x  = self.encoder(x) 
        gen = self.decoder(x)
        return x, gen

def train_autoencoder(df):
    # d
    X = torch.from_numpy(df.T.values)
    # print(X.shape[1])
    dim = int(X.shape[1])

    # Initialize the model, loss_fn, and optimizers
    model = AutoEncoder(dim)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        X = X.to(torch.float32)
        
        _, gen = model(X)
        batch_loss = loss_fn(X, gen)
        
        batch_loss.backward()
        optimizer.step()

    latent_representation = model.encoder(X)
    latent_df = latent_representation.detach().numpy()
    latent_df = pd.DataFrame(latent_df, index = df.columns).T
    
    return latent_df