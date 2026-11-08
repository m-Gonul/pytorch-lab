import numpy as np


def adagrad_step(w, g, st, lr=0.1, eps=1e-10):
    st["G"] = st.get("G", 0) + g * g
    return w - lr * g / (np.sqrt(st["G"]) + eps)


def adamax_step(w, g, st, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] = st.get("t", 0) + 1
    st["m"] = b1 * st.get("m", 0) + (1 - b1) * g
    st["u"] = np.maximum(b2 * st.get("u", 0), np.abs(g))
    return w - lr / (1 - b1 ** t) * st["m"] / (st["u"] + eps)


def nadam_step(w, g, st, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] = st.get("t", 0) + 1
    st["m"] = b1 * st.get("m", 0) + (1 - b1) * g
    st["v"] = b2 * st.get("v", 0) + (1 - b2) * g * g
    mh = b1 * st["m"] / (1 - b1 ** (t + 1)) + (1 - b1) * g / (1 - b1 ** t)
    vh = st["v"] / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(vh) + eps)
