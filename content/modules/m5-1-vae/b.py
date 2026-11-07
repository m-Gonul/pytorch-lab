import math


def kl_gauss(mu, logvar):
    ...


def kl_grad(mu, logvar):
    ...


def reparam(mu, logvar, eps):
    ...


def log_normal(z, mu, logvar):
    ...


def kl_mc(mu, logvar, eps_list):
    ...


print(kl_gauss([1.0], [math.log(0.25)]))    # 0.818
print(reparam([1.0], [math.log(0.25)], [0.4]))   # [1.2]
