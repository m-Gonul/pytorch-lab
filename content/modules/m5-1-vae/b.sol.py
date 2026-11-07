import math


def kl_gauss(mu, logvar):
    return sum(0.5 * (m * m + math.exp(lv) - 1 - lv) for m, lv in zip(mu, logvar))


def kl_grad(mu, logvar):
    return list(mu), [0.5 * (math.exp(lv) - 1) for lv in logvar]


def reparam(mu, logvar, eps):
    return [m + math.exp(0.5 * lv) * e for m, lv, e in zip(mu, logvar, eps)]


def log_normal(z, mu, logvar):
    return sum(-0.5 * (math.log(2 * math.pi) + lv + (x - m) ** 2 / math.exp(lv)) for x, m, lv in zip(z, mu, logvar))


def kl_mc(mu, logvar, eps_list):
    zero = [0.0] * len(mu)
    tot = 0.0
    for eps in eps_list:
        z = reparam(mu, logvar, eps)
        tot += log_normal(z, mu, logvar) - log_normal(z, zero, zero)
    return tot / len(eps_list)


print(kl_gauss([1.0], [math.log(0.25)]))    # 0.818
print(reparam([1.0], [math.log(0.25)], [0.4]))   # [1.2]
