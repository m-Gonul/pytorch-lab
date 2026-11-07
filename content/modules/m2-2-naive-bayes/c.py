import numpy as np
import pandas as pd


def fit_gnb(X, y, C, eps=1e-9):
    ...


def predict_log_proba_gnb(model, X):
    ...


def cond_table(df, feature, target, alpha=1.0):
    ...


rng = np.random.default_rng(0)
y = np.repeat([0, 1], 50)
X = rng.normal(size=(100, 2)) + np.array([[0, 0], [2, 2]])[y]
m = fit_gnb(X, y, 2)
print((predict_log_proba_gnb(m, X).argmax(1) == y).mean())
df = pd.DataFrame({"hava": list("ggyygy"), "oyna": list("EEHEHH")})
print(cond_table(df, "hava", "oyna"))
