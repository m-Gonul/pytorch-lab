import copy
import torch
import torch.nn.functional as F
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / (1 - self.p)


def fit_early_stopping(model, Xtr, ytr, Xva, yva, patience=10, max_epochs=500, lr=1e-2):
    """→ (model [en iyi ağırlıklar yüklü], best_epoch, history)"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, best_state, bad, history = float("inf"), -1, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        loss = F.cross_entropy(model(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = F.cross_entropy(model(Xva), yva).item()
        history.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history
