import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GATMultilabel import GAT

'''
    # Hyperparameters
    input_size = 768
    hidden_size = 250
    num_classes = 90
    learning_rate = 0.001
    batch_size = 250
    num_epochs = 250
    attention_heads = 4

    #Loss and Optimezer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
'''

class MAGNET(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, adj, rnn='lstm'):
        super(BRNN, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.gat1 = GAT(input_size, hidden_size*2, attention_heads)
        self.gat2 = GAT(hidden_size*2, hidden_size*2, attention_heads)
        self.sigmoid = nn.Sigmoid()
    
    # x is text features from sentences representation using BERT with size (SEQ_LEN, 768)
    # feat is label embedding
    # adj is adjacency

    def forward(self, x, feat, adj): # feat.size : (N, 768)

        features, _ = self.lstm(x)
        features = features[:, -1, :].squeeze(1) #features.size : (batch_size, hidden_size*2)

  
        att = self.gat1(feat, adj)
        att = self.gat2(att, adj)
        att = att.transpose(0, 1) #att.size : (hidden_size*2, N)

        out = torch.matmul(features, att)
        out = torch.sigmoid(out) #out.size : (Batch_size, N)

        return out