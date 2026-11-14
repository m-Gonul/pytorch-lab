import math
import random


def transition(s, a, kind, changed):
    # TODO: 6×9 ızgara, başlangıç48, hedef8; değişimden önce/sonra engeller.
    raise NotImplementedError


def select_action(qs, ages, kappa, rng):
    # TODO: Q+κ√τ en büyüğü; eşitlikte rastgele. ε karışımı yok.
    raise NotImplementedError


def run_bonus(kind="shortcut", variant="selection", n_steps=6000, change=3000,
              kappa=.001, seed=0, n_plan=50, alpha=1., epsilon=.1):
    # TODO: plain / plus / selection karşılaştırması; bonus ve gerçek ödülü ayır.
    raise NotImplementedError
