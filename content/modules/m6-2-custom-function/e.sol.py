import torch


class MemEffSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g * s * (1 + x * (1 - s))


def mem_eff_silu(x):
    return MemEffSiLU.apply(x)


x = torch.randn(4, requires_grad=True)
mem_eff_silu(x).sum().backward()
print(x.grad)
