import numpy as np

VOCAB = "0123456789+=\n"


def make_dataset(n, rng):
    a, b = rng.integers(0, 100, (2, n))
    s = a + b
    return np.stack([a // 10, a % 10, np.full(n, 10), b // 10, b % 10, np.full(n, 11),
                     s % 10, (s // 10) % 10, s // 100, np.full(n, 12)], axis=1)


def encode(text):
    lut = np.full(256, -1)
    lut[np.frombuffer(VOCAB.encode(), np.uint8)] = np.arange(len(VOCAB))
    return lut[np.frombuffer(text.encode(), np.uint8)].reshape(-1, 10)


def answer_entropy(D):
    p = np.stack([np.bincount(D[:, j], minlength=13) for j in (6, 7, 8)]) / len(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-(np.where(p > 0, p * np.log(p), 0)).sum())


D = make_dataset(5, np.random.default_rng(0))
print(D)
print(encode("07+45=250\n"))
print(answer_entropy(make_dataset(20000, np.random.default_rng(0))))
