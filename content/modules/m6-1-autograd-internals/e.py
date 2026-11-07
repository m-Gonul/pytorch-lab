import torch


def bf16_saved(fn, *args):
    """fn(*args); saklanan kayan noktalı tensörler bf16'da tutulur."""
    ...


torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256))
x = torch.randn(128, 256)
bf16_saved(mlp, x).pow(2).mean().backward()
print(mlp[0].weight.grad.norm())
