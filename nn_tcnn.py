import torch.nn.functional as F
from torch import nn, optim
from nn_tcnn_convolutions import TCNNConvBlock
from nn_tcnn_classificator import ClassificationHead
from utils import compute_feature_map_size_tcnn
import torch


class TCNN(nn.Module):
    def __init__(self, in_channels, window_length, sensor_channels, filter_size, num_classes, num_filters):
        super().__init__()
        W,H = compute_feature_map_size_tcnn(0,sensor_channels,window_length,filter_size)
        self.embedding_size = W
        latent_size = W*H*num_filters

        self.convolutions = TCNNConvBlock(in_channels, num_filters, filter_size)
        self.classification = ClassificationHead(latent_size, num_classes)



    def forward(self, x):
        reconst = 0 # needed for an easy train loop

        x = self.convolutions.forward(x)
        embedding = x 

        x = torch.flatten(x,1)
        pred = self.classification.forward(x)

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size
