import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def choose_hps(df, approach):
    best_num_clusters = -1
    best_silhouette_score = -1
    
    # Loop through the range of cluster numbers
    for num_clusters in range(5, 21):
        transposed_df = df.transpose()
        # Create the corresponding clustering model based on the approach
        if approach == 'K-Means':
            cluster_model = KMeans(n_clusters=num_clusters)
        elif approach == 'Gaussian mixture model':
            cluster_model = GaussianMixture(n_components=num_clusters)
        elif approach == 'Agglomerative Clustering':
            cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
        else:
            raise ValueError(f"Invalid cluster approach: {approach}")
        
        # Fit the model and predict cluster labels
        cluster_labels = cluster_model.fit_predict(transposed_df)
        
        # Compute the Silhouette Coefficient
        silhouette_avg = silhouette_score(transposed_df, cluster_labels)
        # print("num_clusters:" ,num_clusters, "score:", silhouette_avg)
        # Update the best Silhouette Coefficient and number of clusters if necessary
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
    
    return best_num_clusters



def cluster_approaches(df, num_clusters, cluster_approach='KMeans'):
    # Create the clustering model based on the chosen approach and number of clusters
    if cluster_approach == 'K-Means':
        cluster_model = KMeans(n_clusters= num_clusters)
    elif cluster_approach == 'Gaussian mixture model':
        cluster_model = GaussianMixture(n_components=num_clusters)
    elif cluster_approach == 'Agglomerative Clustering':
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
    else:
        raise ValueError(f"Invalid cluster approach: {cluster_approach}")

    # Fit the clustering model and obtain the cluster labels for each stock
    transposed_df = df.transpose()
    cluster_labels = cluster_model.fit_predict(transposed_df)
    # selected_stocks = []

    # Iterate over each cluster
    '''
    for cluster in range(num_clusters):
        # Get the indices of stocks belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # Calculate the centroid of the current cluster
        cluster_centroid = np.mean(transposed_df.iloc[cluster_indices], axis=0)

        # Calculate the Euclidean distance between each stock in the cluster and the centroid
        distances = cdist(transposed_df.iloc[cluster_indices], [cluster_centroid], metric='euclidean')

        # Find the index of the stock with the minimum distance to the centroid
        representative_stock_index = cluster_indices[np.argmin(distances)]

        # Add the representative stock to the selected stocks list
        selected_stocks.append(transposed_df.index[representative_stock_index])
    '''
    return cluster_labels
