import math
import random


def mlm_mask(tokens, rng, V, mask_id, n_special, p=0.15):
    ...


def nsp_pairs(sentences, rng):
    ...


def bert_params(V, d, L, ff, max_pos=512, types=2):
    ...


def bt_loss(rw, rl):
    ...


def kl_reward(r, logp, logp_ref, beta):
    ...


def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    ...
