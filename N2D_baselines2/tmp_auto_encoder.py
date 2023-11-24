import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        
    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class TmpAutoEncoder(nn.Module):
    def __init__(self, input_size, seq_len):
        super(TmpAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Encoder(input_size, 200),
            nn.BatchNorm1d(seq_len),
            Encoder(200, 100),
            nn.Dropout(0.1),
            Encoder(100, 50)
        )
        
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(seq_len),
            Decoder(50, 100),
            Decoder(100, 200),
            nn.Dropout(0.1),
            Decoder(200, input_size)
        )

            
    def forward(self, x):
        x = x.to('cuda:0')
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        return hidden, output