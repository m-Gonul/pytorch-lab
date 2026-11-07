import numpy as np

VOCAB = "0123456789+=\n"


def make_dataset(n, rng):
    ...


def encode(text):
    ...


def answer_entropy(D):
    ...


D = make_dataset(5, np.random.default_rng(0))
print(D)
print(encode("07+45=250\n"))
print(answer_entropy(make_dataset(20000, np.random.default_rng(0))))
