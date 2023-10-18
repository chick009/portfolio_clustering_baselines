import numpy as np
import torch.nn.functional as F
from N2D import N2D


def evaluate(batch_data, model, total_labels, device = 'cpu'):
    # Get the sampling data 

    # K-Means Clustering (Labels) 
    kmean_labels = total_labels[0]
    gmm_labels = total_labels[1]
    # agglo_labels = total_labels[2]
    # Gaussian Mixture Model (Labels)
    # Agglomerative Clustering (Labels)

    embedding = []
    model = N2D(model, 8)
    
    batch, seq_len, feat = batch_data.shape

    for i in len(batch):
      embedding.append(model.encode(batch_data[i, :, :].to(device)))

    embedding = np.concatenate(embedding)
    manifold = model.manifold(embedding)
    pred = model.cluster(manifold).argmax(1)

    # Evaluating the final prediction similarity along with 
    print(f'KMeans CE Loss: {F.cross_entropy(pred, kmean_labels)}')
    print(f'GMM CE Loss: {F.cross_entropy(pred, gmm_labels)}')
    #print(f'Agglo CE Loss: {F.cross_entropy(pred, agglo_labels)}')

    return pred 