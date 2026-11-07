import time
import torch
import torch.nn.functional as F


def full_softmax_loss(Win, Wout, c, o):
    return F.cross_entropy(Win[c] @ Wout.T, o)


def neg_sampling_loss(Win, Wout, c, o, neg):
    v = Win[c]
    pos = F.logsigmoid((v * Wout[o]).sum(-1))
    negs = F.logsigmoid(-torch.einsum("bd,bkd->bk", v, Wout[neg])).sum(-1)
    return -(pos + negs).mean()
