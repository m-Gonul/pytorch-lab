import io
import torch
from torch.export import export, Dim


def export_dynamic(model, example):
    return export(model, (example,), dynamic_shapes={"x": {0: Dim("batch", min=1, max=1024)}})


def op_names(ep):
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]


def save_load(ep):
    buf = io.BytesIO()
    torch.export.save(ep, buf)
    buf.seek(0)
    return torch.export.load(buf)


class CondModel(torch.nn.Module):
    def forward(self, x):
        return torch.cond(x.sum() > 0, lambda x: x * 2, lambda x: -x, (x,))


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
