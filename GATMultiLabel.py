import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Under development to implement this paper https://arxiv.org/abs/2003.11644
    MAGNET: Multi-Label Text Classification using Attention-based Graph Neural Network

    Step by Step (In my understanding):
       1. All labels embed with BERT, ex:
          'horror' -> BERT -> [v1,...,v768]
          if label have multiple word then it will be added element-wise
          [v1,...,v768] + [v1,..,v768]
       
       2. Then I build the adjacency matrix from Label Correlation with size
          (number of labels * number of labels)
       
       3. In GAT, i have to make Weight Matrix (W) with size 768*num_hidden as Parameter
       4. Label Embedding Matrix (H) will be fed to GAT
       5. To get attention (a) i have to calculate HW multiply HW transpose
          then i fed to ReLU functions [Equation 6],
          then i get matrix with size (num labels * num labels) consisting of aij

       5. Then i make GAT as many as attention_head then i add up all the output
          from GAT element-wise divide by number of attention_head then fed it to
          tanh functions [Equation 7, 8, 9]
    
       6. GAT process finished, ready to fed into next layer or multiply with text features


    Unsolved question:
    1. Where should i put adjacency matrix in GAT?
    2. In Equation 5, what does concatenating mean? In my understanging,
       concatenating mean is combining two arrays according to dimension, ex: a=[1, 2, 3] and
       b=[4, 5, 6] then concat(a,b, dim=0) will be [1,2,3,4,5,6] but Equation 5
       said i have to transpose HjW. So i choose to multiply HiW and HjW
    3 Equation 4 and Equation 7 (multiple attention) have same goal which is to calculate new H but Why 
      Equation 4 and Equation 7 have different activation function
    4. In your paper on table 2 there is Vocab Size, i dont understand why do you have different vocab size
       because Pretrained BERT has fixed Vocab Size right?
    5. How do you embed text using BERT? Do you encode whole sentence or
       word by word then you put it to Embedding Layer?
    
       


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
        super(GAT, self).__init__()
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid) for _ in range(nheads)]

    def forward(self, H, adj):
        
        # i still put adjacency as an input even i am not using it
        sum = 0
        for att in self.attentions:
            sum += att(H, adj)
            
        return torch.tanh(sum/self.nheads) #Equation 7, 8, 9