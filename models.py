import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
  def __init__(self, inp, out):
    super(GraphAttentionLayer, self).__init__()
    self.W = nn.Linear(inp, out, bias=False)
    self.a = nn.Linear(out*2, 1, bias=False)
  
  def forward(self, h, adj):
    Wh = self.W(h)
    attention = self.catneighbour(Wh)*adj.unsqueeze(2)
    attention = self.a(attention).squeeze(2)
    attention = F.leaky_relu(attention, 0.1)
    h_hat = torch.mm(attention, Wh)
    h_hat = F.leaky_relu(h_hat, 0.1)
    
    return h_hat
 
  def catneighbour(self, Wh):
    N = Wh.size(0)
    Whi = Wh.repeat_interleave(N, dim=0)
    Whj = Wh.repeat(N, 1)
    WhiWhj = torch.cat([Whi, Whj], dim=1)
    WhiWhj = WhiWhj.view(N, N, Wh.size(1)*2)
    return WhiWhj
 
class MultiHeadGAT(nn.Module):
  def __init__(self, inp, out, heads, merge=False):
    super(MultiHeadGAT, self).__init__()
    self.merge = merge
    self.attentions = nn.ModuleList([GraphAttentionLayer(inp, out) for _ in range(heads)])
  
  def forward(self, h, adj):
    heads_out = [att(h, adj) for att in self.attentions]
    if self.merge:
      out = torch.cat(heads_out, dim=1)
    else:
      out = torch.stack(heads_out, dim=0).mean(0)
    
    return torch.tanh(out)
 
class GAT(nn.Module):
  def __init__(self, inp, out, heads):
    super(GAT, self).__init__()
    self.gat1 = MultiHeadGAT(inp, out, heads)
    self.gat2 = MultiHeadGAT(out, out, heads)
  
  def forward(self, h, adj):
    out = self.gat1(h, adj)
    out = self.gat2(out, adj)
    return out

class MAGNET(nn.Module):
  def __init__(self, input_size, hidden_size, adjacency, heads=4, dropout=0.5):
    super(MAGNET, self).__init__()
    self.rnntype = rnntype
    if self.rnntype == 'lstm':
      self.rnn = nn.LSTM(input_size, hidden_size,
                         batch_first=True, bidirectional=True)

    self.gat = GAT(input_size, hidden_size*2, heads)
    self.adjacency = nn.Parameter(adjacency)
    self.drop = nn.Dropout(dropout)
 
  def forward(self, features, h):
    
    out, (hidden, cell) = self.rnn(features)

    out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
    out = self.drop(out)
    
    att = self.gat(h, self.adjacency)
    att = self.drop(att)
    att = att.transpose(0, 1)
    
    out = torch.mm(out, att)
 
    return out