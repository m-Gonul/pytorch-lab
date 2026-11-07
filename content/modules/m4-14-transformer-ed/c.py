import numpy as np


def pe(n, d):
    ...


def offset_rotation(k, d):
    ...


def mha(xq, xkv, Wq, Wk, Wv, Wo, h, allow):
    ...


def decoder_masks(tgt_lens, src_lens, T, S):
    ...


def ls_ce(logits, t, eps, pad=0):
    ...
