import random


def dropout_forward(x, p, rng, train=True):
    if not train:
        return list(x), None
    mask = [1 if rng.random() >= p else 0 for _ in x]
    return [v * m / (1 - p) for v, m in zip(x, mask)], mask


def dropout_backward(dy, mask, p):
    return [g * m / (1 - p) for g, m in zip(dy, mask)]


def early_stopping(val_losses, patience, min_delta=0.0):
    best, bad = 0, 0
    for i in range(1, len(val_losses)):
        if val_losses[i] < val_losses[best] - min_delta:
            best, bad = i, 0
        else:
            bad += 1
            if bad >= patience:
                return best, i
    return best, len(val_losses) - 1


print(dropout_forward([2, 4, 6, 8], 0.5, random.Random(0)))
print(early_stopping([0.9, 0.7, 0.6, 0.62, 0.65, 0.61], 2))   # (2, 4)
