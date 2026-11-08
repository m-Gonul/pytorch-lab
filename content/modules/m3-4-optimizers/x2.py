import numpy as np


def adagrad_step(w, g, st, lr=0.1, eps=1e-10):
    ...


def adamax_step(w, g, st, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    ...


def nadam_step(w, g, st, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    ...
