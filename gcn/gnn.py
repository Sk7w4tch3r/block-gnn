import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gcn.layers import GCLayer
from block_sparse import BlockConv

from utils import kernel_to_adj, image_to_adj


class CNNLike():
    def __init__(self, image_size, kernels, biases, fc) -> None:
        
        self.kernels = kernels
        self.conv_biases = biases

        self.fc = fc
        
        self.adj = image_to_adj(np.zeros((image_size, image_size)))
        self.adjs = list(range(len(self.kernels)))
        
        for c in range(len(self.kernels)):
            self.adjs[c] = kernel_to_adj(self.adj, self.kernels[c])


    def forward(self, x):
        x = x.view(x.size(1)*x.size(2)*x.size(3), -1)
        out = []
        for c in range(len(self.kernels)):
            out.append(F.relu(torch.matmul(self.adjs[c], x) + self.conv_biases[c]))
        out = torch.stack(out)
        print(out.shape)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)



class GNN():
    def __init__(self, image_size, in_shape, out_shape, biases, fc) -> None:
        
        self.conv_biases = biases

        self.fc = fc
        
        self.adj = image_to_adj(np.zeros((image_size, image_size)))
        self.weights = nn.Parameter(torch.zeros(len(self.kernels), self.adj.shape[0], self.adj.shape[1]))

        self.register_buffer('adj', self.adj)
        
        for c in range(len(self.kernels)):
            self.adjs[c] = kernel_to_adj(self.adj, self.kernels[c])


    def forward(self, x):
        x = x.view(x.size(1)*x.size(2)*x.size(3), -1)
        out = []
        for c in range(len(self.kernels)):
            
            out.append(F.relu(torch.matmul(self.adjs[c], x) + self.conv_biases[c]))
        out = torch.stack(out)
        print(out.shape)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


    def _mask_tensor(self, i):
        return nn.Parameter(self.adjs[i] * self.adj)
    


class SparseCNN(nn.Module):
    def __init__(self, fin, n_classes) -> None:
        super(SparseCNN, self).__init__()

        self.in_channels = fin
        self.n_classes = n_classes

        self.conv = GCLayer(8, fin, 32, False)
        self.fc = nn.LazyLinear(n_classes)


    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.n_classes) + ')'
    


class BlockSparseCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(BlockSparseCNN, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        image_size = 8

        self.conv = BlockConv(in_channels=in_channels, out_channels=32, kernel_size=3, image_size=image_size, bias=False)
        self.fc = nn.LazyLinear(n_classes, device='cuda')


    def forward(self, x):
    
        x = F.relu(self.conv(x))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.n_classes) + ')'