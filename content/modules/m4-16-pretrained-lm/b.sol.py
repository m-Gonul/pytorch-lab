import math
import random


def mlm_mask(tokens, rng, V, mask_id, n_special, p=0.15):
    inp, lab = [], []
    for t in tokens:
        if t < n_special or rng.random() >= p:
            inp.append(t)
            lab.append(-100)
            continue
        u = rng.random()
        inp.append(mask_id if u < 0.8 else (rng.randrange(n_special, V) if u < 0.9 else t))
        lab.append(t)
    return inp, lab


def nsp_pairs(sentences, rng):
    out = []
    n = len(sentences)
    for i in range(n - 1):
        if rng.random() < 0.5:
            out.append((sentences[i], sentences[i + 1], True))
        else:
            j = rng.choice([k for k in range(n) if k != i + 1])
            out.append((sentences[i], sentences[j], False))
    return out


def bert_params(V, d, L, ff, max_pos=512, types=2):
    layer = 4 * (d * d + d) + 2 * d * ff + ff + d + 4 * d
    return (V + max_pos + types) * d + 2 * d + L * layer + d * d + d


def _sig(x):
    return 1 / (1 + math.exp(-x))


def bt_loss(rw, rl):
    return -math.log(_sig(rw - rl))


def kl_reward(r, logp, logp_ref, beta):
    return r - beta * (logp - logp_ref)


def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    return -math.log(_sig(beta * ((lp_w - ref_w) - (lp_l - ref_l))))
