import math
import random
from collections import Counter, defaultdict


def fit(text, k):
    ...


def log_prob(model, text, k, alpha=1.0, vocab=None):
    ...


def generate(model, seed_text, n, k, seed=0):
    ...


m = fit("abracadabra", 1)
print(m["a"])
