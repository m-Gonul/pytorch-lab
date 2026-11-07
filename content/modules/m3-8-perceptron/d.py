import math
import torch
import torch.nn as nn


def make_data(fn, n, seed):
    ...


def mlp(d_in, d_out, depth, width=64):
    ...


def fit(model, X, Y, steps=2000, lr=3e-3, seed=0):
    ...


def depth_table(fns, depths, n=2000):
    ...


def penultimate_separability(model, X, y):
    ...
