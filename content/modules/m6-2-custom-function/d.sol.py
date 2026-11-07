import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = x.amax(-1, keepdim=True)
        out = (x - m).exp().sum(-1).log() + m.squeeze(-1)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, g):
        x, out = ctx.saved_tensors
        return g[..., None] * (x - out[..., None]).exp()


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


x = torch.tensor([[1.0, 2.0], [1000.0, 1000.0]], requires_grad=True)
y = LogSumExp.apply(x)
y.sum().backward()
print(y, x.grad)
