import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        standard_variance = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-standard_variance, standard_variance)
        if self.bias is not None:
            self.bias.data.uniform_(-standard_variance, standard_variance)
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.zeros_(self.bias)

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, num_features, num_classes, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(num_features, 64)
        self.gc2 = GraphConvolutionLayer(64, 128)
        self.gc3 = GraphConvolutionLayer(128, num_classes)
        self.dropout = nn.Dropout(p = dropout)

    def initialize_weights(self):
        self.gc1.initialize_weights()
        self.gc2.initialize_weights()
        self.gc3.initialize_weights()

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.gc1(x, adjacency_matrix))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, adjacency_matrix))
        x = self.dropout(x)
        x = self.gc3(x, adjacency_matrix)
        return F.log_softmax(x, dim = 1)
