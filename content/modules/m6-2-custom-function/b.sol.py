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
        m = max(xs)
        out = m + math.log(sum(math.exp(x - m) for x in xs))
        ctx.save_for_backward(list(xs), out)
        return out

    @staticmethod
    def backward(ctx, g):
        xs, out = ctx.saved
        return [g * math.exp(x - out) for x in xs]


class RoundSTE(Function):
    @staticmethod
    def forward(ctx, xs):
        return [float(round(x)) for x in xs]

    @staticmethod
    def backward(ctx, g):
        return list(g) if isinstance(g, list) else g


def gradcheck(fn_cls, xs, eps=1e-6, tol=1e-5):
    out, ctx = fn_cls.apply(xs)
    analytic = fn_cls.backward(ctx, 1.0)
    if not isinstance(analytic, list):
        analytic = [analytic] * len(xs)
    err = 0.0
    for i in range(len(xs)):
        p = list(xs); p[i] += eps
        m = list(xs); m[i] -= eps
        fp, fm = fn_cls.apply(p)[0], fn_cls.apply(m)[0]
        if isinstance(fp, list):
            fp, fm = sum(fp), sum(fm)
        num = (fp - fm) / (2 * eps)
        err = max(err, abs(num - analytic[i]))
    return err < tol, err


print(LogSumExp.apply([1.0, 2.0])[0])        # 2.3133
print(gradcheck(LogSumExp, [1.0, 2.0, -0.5]))
