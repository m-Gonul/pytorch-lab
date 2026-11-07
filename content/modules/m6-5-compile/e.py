import torch


def count_breaks(fn, *args):
    ...


def branchy(x):
    out = x * 2
    if out.sum().item() > 0:
        out = out.tanh()
    else:
        out = out.sigmoid()
    return out + 1


def branchy_fixed(x):
    ...


def chain(x):
    ...


compiled_chain = torch.compile(chain)

x = torch.randn(1000)
print(count_breaks(branchy, x), count_breaks(branchy_fixed, x))   # ≥1, 0
