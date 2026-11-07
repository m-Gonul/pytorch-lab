import math


def round_float(x, man_bits, exp_bits):
    ...


class LossScaler:
    def __init__(self, init=2.0 ** 16, growth=2.0, backoff=0.5, interval=3):
        ...

    def update(self, found_inf):
        ...


print(round_float(300.0 ** 2, 10, 5), round_float(1 + 1e-4, 10, 5), round_float(1e-8, 10, 5))   # inf 1.0 0.0
s = LossScaler(8.0)
for inf in [False, False, False, True, False]:
    s.update(inf)
    print(s.scale)
