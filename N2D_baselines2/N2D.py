from sklearn.mixture import GaussianMixture

import torch
from torch import nn

from umap import UMAP

class N2D(nn.Module):
    def __init__(self, encoder, n_cluster):
        super().__init__()
        self.encoder = encoder
        self.n_cluster = n_cluster
        self.umap = None
        self.gmm = None
    
    def forward(self, x):
        print('before encode shape', x.shape)
        x = self.encode(x)
        print('before manifold shape', x.shape)
        x = self.manifold(x)
        print('before cluster shape', x.shape)
        cluster = self.cluster(x)
        return cluster

    def encode(self, x):

        with torch.no_grad():
            x = self.encoder(x)
            if isinstance(x, tuple):
                x = x[0][:, -1, :]
        return x.cpu().numpy()

    def manifold(self, x):
        print("umap shape", x.shape)
        if self.umap is None:
            print('fit the UMAP ...')
            self.umap = UMAP(20, self.n_cluster).fit(x)
        
        x = self.umap.transform(x)
        return x

    def cluster(self, x):
        if self.gmm is None:
            print('fit the GMM ...')
            self.gmm = GaussianMixture(self.n_cluster).fit(x)
        prob = self.gmm.predict_proba(x)
        return prob