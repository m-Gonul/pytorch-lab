import math
import random


def true_grad():
    ...


def grad_samples(n, kind, seed=0):
    ...


print(true_grad(), sum(grad_samples(2000, "V")) / 2000)
