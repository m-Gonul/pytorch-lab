import numpy as np


def mlm_mask_batch(X, rng, V, mask_id, n_special, p=0.15):
    ...


def elmo_mix(H, s, gamma):
    ...


def seq_logprob(logits, y, resp_mask):
    ...


def dpo_batch(pw, pl, rw, rl, beta):
    ...


def fit_reward(Fw, Fl, lr=0.5, iters=500):
    ...
