import numpy as np
import torch
import torch.nn.functional as F
from N2D import N2D

def multi_label_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # Small value to avoid division by zero
    
    # Clip the predictions to a small range for numerical stability
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Calculate the cross-entropy loss for each label
    loss_per_label = - (targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Average the loss across all labels
    loss = np.mean(loss_per_label)
    
    return loss

def evaluate(batch_data, model, total_labels, device = 'cpu'):
    # Get the sampling data 

    # K-Means Clustering (Labels) 
    kmean_labels = total_labels[0]
    gmm_labels = total_labels[1]
    agglo_labels = total_labels[2]
    # Gaussian Mixture Model (Labels)
    # Agglomerative Clustering (Labels)

    embedding = []
    model = N2D(model, 8)
    
    for y in batch_data:
      data = y[0].to(torch.float32)
      data = data.view(data.size(0), -1)
      
      hidden_repr, _ = model.encoder(data.to('cpu'))
      embedding.append(hidden_repr.detach().numpy())

    embedding = np.concatenate(embedding)
    manifold = model.manifold(embedding)
    pred = model.cluster(manifold).argmax(1)
    print(pred)
    # Evaluating the final prediction similarity along with 
    print(f'KMeans CE Loss: {multi_label_cross_entropy_loss(pred, kmean_labels)}')
    print(f'GMM CE Loss: {multi_label_cross_entropy_loss(pred, gmm_labels)}')
    print(f'Agglo CE Loss: {multi_label_cross_entropy_loss(pred, agglo_labels)}')

    return pred 