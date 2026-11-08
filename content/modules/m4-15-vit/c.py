import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def patchify(x, P):
    ...


def unpatchify(z, P, C, H, W):
    ...


def patch_embed_conv(x, Wc, b, P):
    ...


def conv_to_linear(Wc):
    ...


def mean_attention_distance(A, g, P):
    ...
