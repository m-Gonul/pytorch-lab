import torch
import torch.nn.functional as F
from torch import nn


def gcn_propagate(x, edge_index, N):
    ...


class GCN(nn.Module):
    def __init__(self, d_in, hidden, d_out, dropout=0.5):
        super().__init__()
        ...

    def forward(self, x, edge_index):
        ...


def train_node_clf(model, X, ei, y, train_mask, steps=200, lr=0.01):
    ...


ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
print(gcn_propagate(torch.tensor([[1.0], [2.0], [3.0]]), ei, 3))
