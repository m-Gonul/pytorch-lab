import numpy as np


def mlm_mask_batch(X, rng, V, mask_id, n_special, p=0.15):
    sel = (rng.random(X.shape) < p) & (X >= n_special)
    u = rng.random(X.shape)
    inp = np.where(sel & (u < 0.8), mask_id, X)
    rnd = sel & (u >= 0.8) & (u < 0.9)
    inp = np.where(rnd, rng.integers(n_special, V, X.shape), inp)
    return inp, np.where(sel, X, -100)


def elmo_mix(H, s, gamma):
    w = np.exp(s - s.max())
    w /= w.sum()
    return gamma * np.tensordot(w, H, 1)


def seq_logprob(logits, y, resp_mask):
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    tok = np.take_along_axis(lp, y[..., None], -1)[..., 0]
    return (tok * resp_mask).sum(1)


def dpo_batch(pw, pl, rw, rl, beta):
    m = beta * ((pw - rw) - (pl - rl))
    loss = np.logaddexp(0, -m)
    return float(loss.mean()), float((m > 0).mean()), float(m.mean())


def fit_reward(Fw, Fl, lr=0.5, iters=500):
    D = Fw - Fl
    w = np.zeros(D.shape[1])
    for _ in range(iters):
        g = -(D * (1 / (1 + np.exp(D @ w)))[:, None]).mean(0)
        w -= lr * g
    return w
