
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