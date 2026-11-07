import torch


def embedding_bag_mean(weight, ids, offsets):
    """→ (B, d). Döngü yok."""
    ...


W = torch.arange(12.0).view(6, 2)
print(embedding_bag_mean(W, torch.tensor([0, 2, 4, 5, 1]), torch.tensor([0, 2, 2])))   # [[2,3],[0,0],[6.67,7.67]]
