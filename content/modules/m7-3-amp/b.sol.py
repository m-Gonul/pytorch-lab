import math


def round_float(x, man_bits, exp_bits):
    if x == 0 or math.isinf(x) or math.isnan(x):
        return x
    bias = 2 ** (exp_bits - 1) - 1
    max_finite = (2 - 2.0 ** -man_bits) * 2.0 ** bias
    s = -1.0 if x < 0 else 1.0
    a = abs(x)
    m, e2 = math.frexp(a)
    E = max(e2 - 1, 1 - bias)
    q = 2.0 ** (E - man_bits)
    r = round(a / q) * q
    if r > max_finite:
        return s * math.inf
    return s * r


class LossScaler:
    def __init__(self, init=2.0 ** 16, growth=2.0, backoff=0.5, interval=3):
        self.scale, self.growth, self.backoff, self.interval = init, growth, backoff, interval
        self.good = 0

    def update(self, found_inf):
        if found_inf:
            self.scale *= self.backoff
            self.good = 0
        else:
            self.good += 1
            if self.good == self.interval:
                self.scale *= self.growth
                self.good = 0


print(round_float(300.0 ** 2, 10, 5), round_float(1 + 1e-4, 10, 5), round_float(1e-8, 10, 5))   # inf 1.0 0.0
s = LossScaler(8.0)
for inf in [False, False, False, True, False]:
    s.update(inf)
    print(s.scale)
