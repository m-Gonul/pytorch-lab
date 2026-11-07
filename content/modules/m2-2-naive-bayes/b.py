import math
from collections import Counter, defaultdict


def fit_nb(rows, labels, alpha=1.0):
    ...


def predict_proba(model, row):
    ...


def predict(model, row):
    ...


rows = [("g", "z"), ("g", "s"), ("y", "s"), ("y", "z"), ("g", "s"), ("y", "s")]
labels = ["E", "E", "H", "E", "H", "H"]
print(predict_proba(fit_nb(rows, labels, alpha=0.0), ("g", "s")))   # E: 0.4
print(predict_proba(fit_nb(rows, labels, alpha=1.0), ("g", "s")))   # E: 0.4286
