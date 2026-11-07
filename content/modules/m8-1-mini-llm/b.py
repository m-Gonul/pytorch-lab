import math


def lr_schedule(step, warmup, total, max_lr, min_lr):
    ...


def train_flops(N, D):
    ...


def chinchilla_tokens(N):
    ...


def make_line(a, b):
    ...


def parse_answer(s):
    ...


print(lr_schedule(600, 100, 1100, 1e-3, 1e-4))     # 5.5e-4
print(make_line(7, 45), parse_answer("250"))        # 07+45=250 / 52
