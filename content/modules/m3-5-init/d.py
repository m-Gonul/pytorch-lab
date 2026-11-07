import torch
from torch import nn


def init_model(model, scheme):
    ...


def layer_stats(model, X):
    """→ (act_stds, grad_stds): her nn.Linear çıktısı için, katman sırasıyla."""
    ...


def deep_mlp(width=128, depth=20, act=nn.ReLU):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), act()]
    return nn.Sequential(*layers)
