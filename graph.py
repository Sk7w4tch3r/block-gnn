import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, img_size, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.img_size = img_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.correlation = Parameter(torch.FloatTensor(img_size**2, img_size**2))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        zero_vec = -9e15*torch.ones_like(adj)
        self.correlation = torch.where(adj > 0, self.correlation, zero_vec)
        output = torch.matmul(self.correlation, output)
                
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, in_features, n_hid, n_class, dropout, img_size):
        super(GCN, self).__init__()

        self.conv1 = GraphConvolution(in_features, n_hid, img_size)
        self.conv2 = GraphConvolution(n_hid, n_hid, img_size)

        self.dropout = dropout

        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(n_hid, n_class)
        self.fc3 = torch.nn.Linear(n_class, n_class)
        

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        
        # avg pooling
        x = torch.mean(x, dim=1, keepdim=True)

        # x = self.fc1(x)
        # x = F.selu(x)
        x = self.fc2(x)
        # x = F.selu(x)
        # x = self.fc3(x)

        x = F.softmax(x, dim=0)
    
        return x
    
