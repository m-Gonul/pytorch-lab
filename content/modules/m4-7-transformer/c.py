import numpy as np


def layer_norm(x, g, b, eps=1e-5):
    ...


def gelu(x):
    ...


def gpt_forward(idx, P, n_head):
    ...


rng = np.random.default_rng(0)
V, T, d, L = 11, 8, 8, 2
def blk():
    r = lambda *s: rng.normal(size=s) * 0.2
    return dict(ln1_g=np.ones(d), ln1_b=np.zeros(d), attn_w=r(d, 3 * d), attn_b=r(3 * d), proj_w=r(d, d), proj_b=r(d),
                ln2_g=np.ones(d), ln2_b=np.zeros(d), fc_w=r(d, 4 * d), fc_b=r(4 * d), fc2_w=r(4 * d, d), fc2_b=r(d))
P = dict(wte=rng.normal(size=(V, d)) * 0.2, wpe=rng.normal(size=(T, d)) * 0.2, blocks=[blk(), blk()], lnf_g=np.ones(d), lnf_b=np.zeros(d))
print(gpt_forward(np.array([[1, 2, 3, 4]]), P, n_head=2).shape)   # (1, 4, 11)
