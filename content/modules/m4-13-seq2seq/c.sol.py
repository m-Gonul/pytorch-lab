import numpy as np


def masked_softmax(scores, mask):
    z = np.where(mask, scores, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(-1, keepdims=True)


def additive_attention(s, H, W1, W2, v, mask):
    e = np.tanh(H @ W1 + (s @ W2)[:, None]) @ v
    a = masked_softmax(e, mask)
    return a, np.einsum("bt,btd->bd", a, H)


def luong_attention(s, H, mask, W=None):
    q = s if W is None else s @ W
    a = masked_softmax(np.einsum("bd,btd->bt", q, H), mask)
    return a, np.einsum("bt,btd->bd", a, H)


def masked_ce(logits, targets, pad=0):
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    nll = -np.take_along_axis(lp, targets[..., None], -1)[..., 0]
    m = targets != pad
    return float((nll * m).sum() / m.sum())
