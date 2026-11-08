import torch
import torch.nn.functional as F


def dpo_loss(pw, pl, rw, rl, beta):
    return -F.logsigmoid(beta * ((pw - rw) - (pl - rl))).mean()


def optimal_logp(lref, r, beta):
    return torch.log_softmax(lref + r / beta, 0)


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum()
