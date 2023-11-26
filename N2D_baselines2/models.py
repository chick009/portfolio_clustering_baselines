import torch.nn as nn
import torch
from sklearn.cluster import AgglomerativeClustering

class TAE_encoder(nn.Module):
    def __init__(self, input_shape, filter_1, filter_lstm, pooling):
        super().__init__()

        # Layer 1 & 2: CNN Filter + Pooling Operations 
        self.filter_1 = filter_1
        self.pooling = pooling
        self.input_shape = input_shape
        self.feats = input_shape[-1]

        # Layer 3 & 4: Bi-LSTM Layers
        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.n_hidden = filter_lstm[1]
        
        kernel_height, kernel_width = 10, 1
        stride_height, stride_width = self.pooling, 1

        padding_height = ((input_shape[1] - 1) * stride_height + kernel_height - input_shape[1]) // 2 + 1
        print("1", padding_height)
        self.conv_layer = nn.Conv2d(
                in_channels = self.feats,  # To be changed, 
                out_channels = self.filter_1,
                kernel_size = (10, 1), 
                stride = (self.pooling, 1), # To be changed ,
                padding = (int(padding_height), 0)
        )
        
        # For layer 3 & 4:
        self.lstm_1 = nn.LSTM(
            input_size = 50,
            hidden_size = self.hidden_lstm_1,
            batch_first = True,
            bidirectional = True 
        )

        self.lstm_2 = nn.LSTM(
            input_size = 50,
            hidden_size = self.hidden_lstm_2,
            batch_first = True,
            bidirectional = True 
        )
    
    def forward(self, x):
        # Inputs: (B x S x F)

        # Permute to shape of (B x F x S)
        x = x.permute((0, 2, 1))

        # Expand dimension to (B x F x S x 1)
        x = torch.unsqueeze(x, dim = 3)
 
        print(x.shape)
        # Outputs: (B x S x 1 x filter_1)
        out_cnn = self.conv_layer(x)
        print(out_cnn.shape)
        # Outputs: (B x S x filter_1)
        out_cnn = out_cnn.view(out_cnn.shape[0],  out_cnn.shape[2], out_cnn.shape[1])
        print(out_cnn.shape)
        # Outputs = (B x S x 2 * F) -> meaning returning sequence
        out_lstm1, _ = self.lstm_1(out_cnn)
        print("outLstm", out_lstm1.shape)
        # Outputs = (B x S x Hidden Units)        
        out_lstm1 = torch.sum(
            out_lstm1.view(
                out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1
            ),
            dim = 2
        )

        # Outputs = (B x S x 1)
        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(
                features.shape[0], features.shape[1], 2, self.hidden_lstm_2
            ),
            dim = 2,
        )
        print(features.shape)

        return features

class TAE_decoder(nn.Module):
    def __init__(self, input_shape, pooling, n_hidden):
        super().__init__()
        self.pooling = pooling
        self.n_hidden = n_hidden
        self.input_size = input_shape
        self.feats = input_shape[-1]

        kernel_height, kernel_width = 10, 1
        stride_height, stride_width = self.pooling, 1

        padding_height = ((input_shape[1] - 1) * stride_height + kernel_height - input_shape[1]) // 2
        
        self.deconv_layer = nn.ConvTranspose2d(
            in_channels= self.n_hidden,
            out_channels= self.feats, # The Hidden Units
            kernel_size= (10, 1),
            stride= (self.pooling, 1),
            padding= (int(padding_height), 0)
        )
        

    def forward(self, x):
        # Inputs Shape (B x S x Hidden Units)
        x = x.permute((0, 2, 1))
        # Outputs: (B x Hidden Units x S x 1)
        x = x.unsqueeze(3)

        # Output Shape (B x F x S x 1)
        out_deconv = self.deconv_layer(x)[:, :, :self.input_size[1], :]
        print("out", out_deconv.shape)
        # Output Shape (B x S x F x 1)
        out_deconv = out_deconv.permute((0, 2, 1, 3))
        # print("out deconv shape", out_deconv.shape)

        return out_deconv
    

class TAE(nn.Module):
    def __init__(self, input_shape, pooling = 8, filter_1 = 50, filter_lstm = [50, 1]):
        super().__init__()

        # Set the pooling size
        self.pooling = pooling
        self.filter_1 = filter_1
        self.filter_lstm = filter_lstm
        self.n_hidden = filter_lstm[1]
        self.input_shape = input_shape

        self.tae_encoder = TAE_encoder(
            input_shape = self.input_shape,
            filter_1 = self.filter_1,
            filter_lstm = self.filter_lstm,
            pooling = self.pooling
        )

        self.tae_decoder = TAE_decoder(
            input_shape = self.input_shape,
            pooling = self.pooling,
            n_hidden = self.n_hidden 
        )

    def forward(self, x):
        # Outputs: features.squeeze(2) generates latent representation for each of the feature map
        # Output Shape: (B x S x 1)
        features = self.tae_encoder(x)
        
        # Output Shape (B x S x F)
        out_deconv = self.tae_decoder(features)

        return features.squeeze(2), out_deconv
    
class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args):
        super().__init__()

        ## init with the pretrained autoencoder model
        self.tae = TAE(args)
        self.tae.load_state_dict(
            torch.load(args.path_weights_ae, map_location=args.device)
        )

        ## clustering model
        self.alpha_ = args.alpha
        self.centr_size = args.n_hidden
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity

        
    def compute_similarity(z, centroids, similarity="EUC"):
        """
        Function that compute distance between a latent vector z and the clusters centroids.

        similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                    for Complexity Invariant Similarity.
        z shape : (batch_size, n_hidden)
        centroids shape : (n_clusters, n_hidden)
        output : (batch_size , n_clusters)
        """
        n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]
        bs = z.shape[0]

        if similarity == "CID":
            CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
            CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
            z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
            mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
            CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
            CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
            CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
            return torch.transpose(mse, 0, 1) * CF

        elif similarity == "EUC":
            z = z.expand((n_clusters, bs, n_hidden))
            mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
            return torch.transpose(mse, 0, 1)

        elif similarity == "COR":
            std_z = (
                torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
            )  ## (bs,n_clusters)
            mean_z = (
                torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
            )  ## (bs,n_clusters)
            std_cen = (
                torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
            )  ## (bs,n_clusters)
            mean_cen = (
                torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
            )  ## (bs,n_clusters)
            ## covariance
            z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
            cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
            prod_expec = torch.mean(
                z_expand * cen_expand, dim=2
            )  ## (bs , n_clusters)
            pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        
        return torch.sqrt(2 * (1 - pearson_corr))
    
    def init_centroids(self, x):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z, _ = self.tae(x.squeeze().unsqueeze(1).detach())
        z_np = z.detach().cpu()
        assignements = AgglomerativeClustering(
            n_clusters=2, linkage="complete", affinity="precomputed"
        ).fit_predict(
            compute_similarity(z_np, z_np, similarity=self.similarity)
        )

        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )

        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

        self.centroids = nn.Parameter(centroids_)

    def forward(self, x):

        z, x_reconstr = self.tae(x)
        z_np = z.detach().cpu()

        similarity = self.compute_similarity(
            z, self.centroids, similarity=self.similarity
        )

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        return z, x_reconstr, Q, P