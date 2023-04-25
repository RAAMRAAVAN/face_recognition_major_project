# Layers: Krishanu
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import glob
import os

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., seed=0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        torch.manual_seed(10)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(10)
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, adj):
        #  the outputs are scaled by a factor of 1/1-p  during training. This means that during evaluation the module simply computes an identity function
        input = F.dropout(input, self.dropout, self.training)
        # Performs a matrix multiplication of the matrices input and mat2.
        support = torch.mm(input.float(), self.weight.float())
        # Performs a matrix multiplication of the sparse matrix mat1 and the (sparse or strided) matrix mat2
        output = torch.spmm(adj, support.float())
        output = self.act(output)
        return output

    def __repr__(self):
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
