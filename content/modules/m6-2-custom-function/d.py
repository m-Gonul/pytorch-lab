import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


x = torch.tensor([[1.0, 2.0], [1000.0, 1000.0]], requires_grad=True)
y = LogSumExp.apply(x)
y.sum().backward()
print(y, x.grad)
