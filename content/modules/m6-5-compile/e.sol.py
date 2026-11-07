import torch


def count_breaks(fn, *args):
    return torch._dynamo.explain(fn)(*args).graph_break_count


def branchy(x):
    out = x * 2
    if out.sum().item() > 0:
        out = out.tanh()
    else:
        out = out.sigmoid()
    return out + 1


def branchy_fixed(x):
    out = x * 2
    return torch.where(out.sum() > 0, out.tanh(), out.sigmoid()) + 1


def chain(x):
    y = x * 1.01 + 0.1
    y = y * 0.99 - 0.2
    y = y * y + 1.0
    return y * 0.5 - x


compiled_chain = torch.compile(chain)

x = torch.randn(1000)
print(count_breaks(branchy, x), count_breaks(branchy_fixed, x))   # ≥1, 0
