import torch
from torch.utils.checkpoint import checkpoint


def param_bytes(model):
    ...


def optimizer_state_bytes(opt):
    ...


def saved_bytes(fn):
    ...


def run_checkpointed(blocks, x, k):
    ...


torch.manual_seed(0)
blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh()) for _ in range(16))
x = torch.randn(64, 256, requires_grad=True)
print(param_bytes(blocks))
print(saved_bytes(lambda: run_checkpointed(blocks, x, 4))[1])   # 4 segment girdisi × 65536
