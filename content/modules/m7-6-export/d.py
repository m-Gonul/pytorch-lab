import io
import torch
from torch.export import export, Dim


def export_dynamic(model, example):
    ...


def op_names(ep):
    ...


def save_load(ep):
    ...


class CondModel(torch.nn.Module):
    def forward(self, x):
        ...


class BadModel(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return -x


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


ep = export_dynamic(MLP().eval(), torch.randn(4, 16))
print(op_names(ep), ep.module()(torch.randn(7, 16)).shape)
