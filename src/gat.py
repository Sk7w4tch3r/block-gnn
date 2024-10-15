import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Weight
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        self.weight2 = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
                        
    def forward(self, x, adj):
        batch_size = x.size()[0]
        node_count = x.size()[1]
        
        x = x.reshape(batch_size * node_count, x.size()[2])
        x = torch.mm(x, self.weight)
        x = x.reshape(batch_size, node_count, self.weight.size()[-1])
        
        # Attention score
        attention_input = torch.cat([x.repeat(1, 1, node_count).view(batch_size, node_count * node_count, -1), 
                                     x.repeat(1, node_count, 1)], dim=2).view(batch_size, node_count, -1, 2 * self.out_features)
        e = F.relu(torch.matmul(attention_input, self.weight2).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        x = torch.bmm(attention, x)
        
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    



class GAT(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, device):
        super(GAT, self).__init__()

        self.n_layer = n_layer
        # no dropout
        self.dropout = 0.0
        
        # Graph attention layer
        self.graph_attention_layers = []
        for i in range(self.n_layer):
          self.graph_attention_layers.append(GraphAttentionLayer(n_feat, agg_hidden, self.dropout, device))
                    
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden*n_layer, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
        
    def forward(self, x, adj):
        
        
        # Graph attention layer
        x = torch.cat([F.relu(att(x, adj)) for att in self.graph_attention_layers], dim=2)

        # Readout
        x = torch.mean(x, dim=1)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        
        return x

    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_attention_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers