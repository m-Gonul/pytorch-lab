import math


def lr_schedule(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def train_flops(N, D):
    return 6 * N * D


def chinchilla_tokens(N):
    return 20 * N


def make_line(a, b):
    return f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}\n"


def parse_answer(s):
    return int(s[::-1])


print(lr_schedule(600, 100, 1100, 1e-3, 1e-4))     # 5.5e-4
print(make_line(7, 45), parse_answer("250"))        # 07+45=250 / 52
