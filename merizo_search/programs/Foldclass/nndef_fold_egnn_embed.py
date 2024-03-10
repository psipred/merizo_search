import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log
#from my_egnn import EGNN
from .my_egnn_nocoords import EGNN


# Inject some information about the relative or absolute sequence number
class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=3000, learned=True):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        if learned:
            self.pe = torch.nn.Parameter(pe)
        else:
            self.register_buffer('pe', pe)

    def forward(self, x):

        x = self.pe[:, :x.size(1), :]
        return self.dropout(x)


# EGNN Module
class FoldClassNet(nn.Module):
    def __init__(self,width):
        super().__init__()

        self.width = width

        self.posenc_as = PositionalEncoder(width, learned=False)

        layers = []

        for rep in range(2):
            layer = EGNN(dim=width, m_dim=width*2, init_eps=1e-3)
            layers.append(layer)

        self.encode_ca_egnn = nn.Sequential(*layers)
        
    def forward(self, x):

        nres = x.size(1)

        seq_feats = self.posenc_as(x)

        #print(x.size(), seq_feats.size())
        out_feats = self.encode_ca_egnn((seq_feats, x, None))[0]

        #print(out_feats.size())

        embed = out_feats.mean(dim=1)
        return embed
