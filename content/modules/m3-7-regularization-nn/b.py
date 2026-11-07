import random


def dropout_forward(x, p, rng, train=True):
    ...


def dropout_backward(dy, mask, p):
    ...


def early_stopping(val_losses, patience, min_delta=0.0):
    ...


print(dropout_forward([2, 4, 6, 8], 0.5, random.Random(0)))
print(early_stopping([0.9, 0.7, 0.6, 0.62, 0.65, 0.61], 2))   # (2, 4)
