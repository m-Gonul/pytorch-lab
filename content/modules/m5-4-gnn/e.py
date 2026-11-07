import torch


def propagate(x, edge_index, N):
    loops = torch.arange(N)
    src, dst = torch.cat([edge_index[0], loops]), torch.cat([edge_index[1], loops])
    deg = torch.zeros(N, dtype=x.dtype).index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype))
    return torch.zeros_like(x).index_add_(0, dst, x[src] * (deg[src] * deg[dst]).rsqrt()[:, None])


# ---- senin kodun ----

def collate(graphs):
    """[(x_i, ei_i)] → (X, EI, batch). Graflar üzerinde tek seviye döngü serbest."""
    ...


def readout(x, batch, B):
    """→ (B, 2d): [ortalama, max]. Döngü yok."""
    ...


g1 = (torch.tensor([[1.0], [2.0]]), torch.tensor([[0, 1], [1, 0]]))
g2 = (torch.tensor([[5.0], [0.0], [1.0]]), torch.tensor([[0, 1], [1, 0]]))
X, EI, batch = collate([g1, g2])
print(EI, batch)                       # kenarlar [[0,1,2,3],[1,0,3,2]], batch [0,0,1,1,1]
print(readout(X, batch, 2))            # [[1.5, 2], [2, 5]]
