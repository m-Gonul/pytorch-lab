import logging
import os
import tempfile
import torch
from torch.export import export, Dim

torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


def compile_package(ep, path):
    ...


def load_package(path):
    ...


torch.manual_seed(0)
m = MLP().eval()
ep = export(m, (torch.randn(4, 64),), dynamic_shapes={"x": {0: Dim("batch", min=1, max=1024)}})
path = compile_package(ep, os.path.join(tempfile.mkdtemp(), "mlp.pt2"))
run = load_package(path)
print((run(torch.randn(9, 64)) - m(torch.randn(9, 64))).shape)
