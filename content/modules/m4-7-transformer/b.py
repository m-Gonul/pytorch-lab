import math
import random


def block_params(d):
    ...


def gpt_params(V, T, d, n_layer):
    ...


def probs(logits, temperature=1.0, top_k=None):
    ...


def sample(logits, u, temperature=1.0, top_k=None):
    ...


def generate(next_logits, prompt, n, temperature=1.0, top_k=None, rng=None):
    ...


print(gpt_params(50257, 1024, 768, 12))        # 124439808
print(probs([2, 1, 0], top_k=2))               # [0.731, 0.269, 0]
bigram = lambda toks: [3.0 if i == (toks[-1] + 1) % 5 else 0.0 for i in range(5)]
print(generate(bigram, [0], 8, temperature=0.7, rng=random.Random(0)))
