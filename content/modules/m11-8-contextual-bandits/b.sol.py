import math
import random


class BayesLinReg:
    def __init__(self, d, lam=1.0, sigma=1.0):
        self.d, self.s2 = d, sigma ** 2
        self.Sigma = [[lam ** 2 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.b = [0.0] * d

    def _Sx(self, x):
        return [sum(self.Sigma[i][j] * x[j] for j in range(self.d)) for i in range(self.d)]

    def update(self, x, r):
        Sx = self._Sx(x)
        den = self.s2 + sum(x[i] * Sx[i] for i in range(self.d))
        for i in range(self.d):
            for j in range(self.d):
                self.Sigma[i][j] -= Sx[i] * Sx[j] / den
            self.b[i] += x[i] * r / self.s2

    def mean(self, x):
        Sb = [sum(self.Sigma[i][j] * self.b[j] for j in range(self.d)) for i in range(self.d)]
        return sum(x[i] * Sb[i] for i in range(self.d))

    def var(self, x):
        Sx = self._Sx(x)
        return sum(x[i] * Sx[i] for i in range(self.d))


def news(T, policy, seed=0, c=1.0):
    rng = random.Random(seed)
    K, d = 5, 5
    theta = [[0.5 * rng.random() for _ in range(d)] for _ in range(K)]
    models = [BayesLinReg(d, 1.0, 0.5) for _ in range(K)]
    n, s = [0] * K, [0.0] * K
    clicks = 0
    for t in range(1, T + 1):
        e = [rng.expovariate(1.0) for _ in range(d)]
        tot = sum(e)
        x = [v / tot for v in e]
        if policy == "linucb":
            a = max(range(K), key=lambda k: models[k].mean(x) + c * math.sqrt(models[k].var(x)))
        elif policy == "ucb1":
            a = t - 1 if t <= K else max(range(K), key=lambda k: s[k] / n[k] + math.sqrt(2 * math.log(t) / n[k]))
        else:
            a = rng.randrange(K)
        p = sum(theta[a][j] * x[j] for j in range(d))
        r = 1 if rng.random() < p else 0
        models[a].update(x, r)
        n[a] += 1
        s[a] += r
        clicks += r
    return clicks


print(news(2000, "linucb"), news(2000, "random"))
