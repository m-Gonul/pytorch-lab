import numpy as np


def make_task(S, b, seed=800001):
    # TODO: b farklı ardıl ve dal başına N(0,1) sabit beklenen ödül.
    raise NotImplementedError


def expected_backup(Q, next_states, rewards, s, a):
    # TODO: .1 terminal olasılığında ödül0; γ1.
    raise NotImplementedError


def evaluate_policy(Q, next_states, rewards, tol=1e-11):
    # TODO: sabit açgözlü politika için V=r+.9 P V; V ve Bellman kalanı dön.
    raise NotImplementedError


def learn(next_states, rewards, steps, mode="trajectory", seed=1700001, points=None):
    # TODO: inplace döngüsel düzgün tarama veya ε.1 yörünge; ikisi de expected backup.
    raise NotImplementedError


def experiment(S=10000, b=1, runs=200, steps=200000, seed=800001, points=None):
    # TODO: eşlenmiş 200 model, ortalama/standart hata/gerçek yedek sayısı.
    raise NotImplementedError
