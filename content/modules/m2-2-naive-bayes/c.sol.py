import numpy as np
import pandas as pd


def fit_gnb(X, y, C, eps=1e-9):
    X = np.asarray(X, float)
    Y = np.eye(C)[y]                                 # (n, C)
    cnt = Y.sum(0)
    mu = Y.T @ X / cnt[:, None]
    var = Y.T @ X ** 2 / cnt[:, None] - mu ** 2 + eps
    return np.log(cnt / len(y)), mu, var


def predict_log_proba_gnb(model, X):
    log_prior, mu, var = model
    D = np.asarray(X, float)[:, None, :] - mu[None]  # (n, C, d)
    ll = -0.5 * (np.log(2 * np.pi * var)[None] + D ** 2 / var[None]).sum(2) + log_prior
    m = ll.max(1, keepdims=True)
    return ll - (m + np.log(np.exp(ll - m).sum(1, keepdims=True)))


def cond_table(df, feature, target, alpha=1.0):
    t = pd.crosstab(df[target], df[feature]) + alpha
    return t.div(t.sum(axis=1), axis=0)


rng = np.random.default_rng(0)
y = np.repeat([0, 1], 50)
X = rng.normal(size=(100, 2)) + np.array([[0, 0], [2, 2]])[y]
m = fit_gnb(X, y, 2)
print((predict_log_proba_gnb(m, X).argmax(1) == y).mean())
df = pd.DataFrame({"hava": list("ggyygy"), "oyna": list("EEHEHH")})
print(cond_table(df, "hava", "oyna"))
