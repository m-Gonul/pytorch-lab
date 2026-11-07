import torch
import torch.nn.functional as F
from torch import nn


def gcn_propagate(x, edge_index, N):
    loops = torch.arange(N)
    src = torch.cat([edge_index[0], loops])
    dst = torch.cat([edge_index[1], loops])
    deg = torch.zeros(N, dtype=x.dtype).index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype))
    w = (deg[src] * deg[dst]).rsqrt()
    return torch.zeros(N, x.shape[1], dtype=x.dtype).index_add_(0, dst, x[src] * w[:, None])


class GCN(nn.Module):
    def __init__(self, d_in, hidden, d_out, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(d_in, hidden)
        self.lin2 = nn.Linear(hidden, d_out)
        self.p = dropout

    def forward(self, x, edge_index):
        N = x.shape[0]
        h = F.relu(gcn_propagate(self.lin1(x), edge_index, N))
        h = F.dropout(h, self.p, self.training)
        return gcn_propagate(self.lin2(h), edge_index, N)


def train_node_clf(model, X, ei, y, train_mask, steps=200, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(steps):
        loss = F.cross_entropy(model(X, ei)[train_mask], y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
print(gcn_propagate(torch.tensor([[1.0], [2.0], [3.0]]), ei, 3))
