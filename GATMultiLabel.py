import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Under development to replicate this paper https://arxiv.org/abs/2003.11644
    MAGNET: Multi-Label Text Classification using Attention-based Graph Neural Network
'''
class GraphAttentionLayer(nn.Module):
 
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.uniform_(self.W.data, a=0.0, b=1.0)

    def forward(self, H, adj):
        HW = torch.mm(H, self.W) #Calculate HW
        attention = torch.relu(torch.mm(HW, HW.transpose(0, 1))) #Equation 6
        h_prime = torch.relu(torch.matmul(attention, HW)) #Equation 4
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid) for _ in range(nheads)]

    def forward(self, x, adj):
        
        sum = 0
        for att in self.attentions:
            sum += att(x, adj)
            
        return torch.tanh(sum/self.nheads)