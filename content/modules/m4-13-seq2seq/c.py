import numpy as np


def masked_softmax(scores, mask):
    ...


def additive_attention(s, H, W1, W2, v, mask):
    ...


def luong_attention(s, H, mask, W=None):
    ...


def masked_ce(logits, targets, pad=0):
    ...
