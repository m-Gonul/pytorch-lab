import numpy as np
import torch


def gaussian_otf(N, s):
    ...


def degrade(x, H, sigma, g):
    ...


def apply(G, y):
    ...


def wiener(H, K):
    ...


def learn_filter(train, H, sigma, steps=600, batch=16, seed=0):
    ...


print(gaussian_otf(8, 1.0).shape)
