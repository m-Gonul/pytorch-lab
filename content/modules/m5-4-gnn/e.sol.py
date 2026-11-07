import torch


def propagate(x, edge_index, N):
    loops = torch.arange(N)
    src, dst = torch.cat([edge_index[0], loops]), torch.cat([edge_index[1], loops])
    deg = torch.zeros(N, dtype=x.dtype).index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype))
    return torch.zeros_like(x).index_add_(0, dst, x[src] * (deg[src] * deg[dst]).rsqrt()[:, None])


# ---- senin kodun ----

def collate(graphs):
    """[(x_i, ei_i)] → (X, EI, batch). Graflar üzerinde tek seviye döngü serbest."""
    sizes = torch.tensor([x.shape[0] for x, _ in graphs])
    offsets = torch.cumsum(sizes, 0) - sizes
    n_edges = torch.tensor([ei.shape[1] for _, ei in graphs])
    X = torch.cat([x for x, _ in graphs])
    EI = torch.cat([ei for _, ei in graphs], dim=1) + torch.repeat_interleave(offsets, n_edges)
    batch = torch.repeat_interleave(torch.arange(len(graphs)), sizes)
    return X, EI, batch


def readout(x, batch, B):
    """→ (B, 2d): [ortalama, max]. Döngü yok."""
    d = x.shape[1]
    cnt = torch.zeros(B, dtype=x.dtype).index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    mean = torch.zeros(B, d, dtype=x.dtype).index_add_(0, batch, x) / cnt.clamp_min(1)[:, None]
    idx = batch[:, None].expand(-1, d)
    mx = torch.full((B, d), float("-inf"), dtype=x.dtype).scatter_reduce(0, idx, x, "amax", include_self=True)
    return torch.cat([mean, mx], dim=1)


g1 = (torch.tensor([[1.0], [2.0]]), torch.tensor([[0, 1], [1, 0]]))
g2 = (torch.tensor([[5.0], [0.0], [1.0]]), torch.tensor([[0, 1], [1, 0]]))
X, EI, batch = collate([g1, g2])
print(EI, batch)                       # kenarlar [[0,1,2,3],[1,0,3,2]], batch [0,0,1,1,1]
print(readout(X, batch, 2))            # [[1.5, 2], [2, 5]]
