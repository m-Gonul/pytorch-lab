import math


class Ctx:
    def __init__(self):
        self.saved = ()

    def save_for_backward(self, *xs):
        self.saved = xs


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Ctx()
        return cls.forward(ctx, *args), ctx


# ---- senin kodun ----

class LogSumExp(Function):
    @staticmethod
    def forward(ctx, xs):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


class RoundSTE(Function):
    @staticmethod
    def forward(ctx, xs):
        ...

    @staticmethod
    def backward(ctx, g):
        ...


def gradcheck(fn_cls, xs, eps=1e-6, tol=1e-5):
    ...


print(LogSumExp.apply([1.0, 2.0])[0])        # 2.3133
print(gradcheck(LogSumExp, [1.0, 2.0, -0.5]))
