import torch


def embedding_bag_mean(weight, ids, offsets):
    """→ (B, d). Döngü yok."""
    B = offsets.shape[0]
    lengths = torch.diff(offsets, append=torch.tensor([ids.shape[0]], dtype=offsets.dtype))
    seg = torch.repeat_interleave(torch.arange(B), lengths)
    sums = torch.zeros(B, weight.shape[1], dtype=weight.dtype).index_add_(0, seg, weight[ids])
    return sums / lengths.clamp_min(1)[:, None].to(weight.dtype)


W = torch.arange(12.0).view(6, 2)
print(embedding_bag_mean(W, torch.tensor([0, 2, 4, 5, 1]), torch.tensor([0, 2, 2])))   # [[2,3],[0,0],[6.67,7.67]]
