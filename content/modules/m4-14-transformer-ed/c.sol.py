import numpy as np


def _freq(d):
    return 10000.0 ** (-np.arange(0, d, 2) / d)


def pe(n, d):
    a = np.arange(n)[:, None] * _freq(d)[None]
    P = np.zeros((n, d))
    P[:, 0::2] = np.sin(a)
    P[:, 1::2] = np.cos(a)
    return P


def offset_rotation(k, d):
    w = k * _freq(d)
    R = np.zeros((d, d))
    i = np.arange(0, d, 2)
    R[i, i] = np.cos(w)
    R[i, i + 1] = np.sin(w)
    R[i + 1, i] = -np.sin(w)
    R[i + 1, i + 1] = np.cos(w)
    return R


def mha(xq, xkv, Wq, Wk, Wv, Wo, h, allow):
    B, T, d = xq.shape
    S = xkv.shape[1]
    dk = d // h
    q = (xq @ Wq).reshape(B, T, h, dk).transpose(0, 2, 1, 3)
    k = (xkv @ Wk).reshape(B, S, h, dk).transpose(0, 2, 1, 3)
    v = (xkv @ Wv).reshape(B, S, h, dk).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
    s = np.where(allow[:, None], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(-1, keepdims=True)
    return (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo


def decoder_masks(tgt_lens, src_lens, T, S):
    tk = np.arange(T)[None] < np.asarray(tgt_lens)[:, None]
    sk = np.arange(S)[None] < np.asarray(src_lens)[:, None]
    causal = np.tril(np.ones((T, T), bool))
    return causal[None] & tk[:, None, :], np.broadcast_to(sk[:, None, :], (len(src_lens), T, S)).copy()


def ls_ce(logits, t, eps, pad=0):
    V = logits.shape[-1]
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    q = np.full(lp.shape, eps / V)
    np.put_along_axis(q, t[..., None], 1 - eps + eps / V, -1)
    m = t != pad
    return float(((-(q * lp).sum(-1)) * m).sum() / m.sum())
