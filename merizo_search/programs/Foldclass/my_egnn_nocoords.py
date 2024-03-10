import torch
from torch import nn, einsum, broadcast_tensors

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Method described in https://arxiv.org/pdf/2102.09844.pdf
# Code derived from implementation at https://github.com/lucidrains/egnn-pytorch

class EGNN(nn.Module):
    def __init__(self, dim, edge_dim = 0, m_dim = 16, radius = 10.0, init_eps = 1e-3):
        super().__init__()

        edge_input_dim = 2 * dim + edge_dim + 1

        self.radius = radius

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            nn.SiLU()
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(self, inputs):
        feats, coors, edges = inputs
        b, n, d = list(feats.size())

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        dist = torch.linalg.norm(rel_coors, dim = -1, keepdim = True)

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_j = rearrange(feats, 'b j d -> b () j d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        #print(feats, dist)
        #print(self.radius)
        #edge_input = torch.cat((feats_i, feats_j, torch.clip(dist/self.radius, max=1.0)), dim = -1)
        edge_input = torch.cat((feats_i, feats_j, dist*dist), dim = -1)
        
        if edges is not None:
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)
        m_ij = m_ij * self.edge_gate(m_ij)

        #m_ij = self.edge_mlp(edge_input) * (dist < self.radius)
        #print(m_ij.size())

        m_i = m_ij.sum(dim = -2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return (node_out, coors, edges)
