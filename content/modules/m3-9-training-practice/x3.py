import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier


def shift_augment(X):
    ...


def few_shot_accuracy(augment, seed, n_per_class=3):
    ...


print(few_shot_accuracy(False, 0), few_shot_accuracy(True, 0))
