import torch


class MemEffSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


def mem_eff_silu(x):
    return MemEffSiLU.apply(x)


x = torch.randn(4, requires_grad=True)
mem_eff_silu(x).sum().backward()
print(x.grad)
