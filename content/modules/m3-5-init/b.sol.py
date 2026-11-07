import math

C = math.sqrt(2 / math.pi)


def relu(x): return x if x > 0 else 0.0
def d_relu(x): return 1.0 if x > 0 else 0.0
def leaky_relu(x, a=0.01): return x if x > 0 else a * x
def d_leaky_relu(x, a=0.01): return 1.0 if x > 0 else a


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x): return math.tanh(x)
def d_tanh(x): return 1 - math.tanh(x) ** 2


def gelu(x):
    return 0.5 * x * (1 + math.tanh(C * (x + 0.044715 * x ** 3)))


def d_gelu(x):
    u = C * (x + 0.044715 * x ** 3)
    t = math.tanh(u)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C * (1 + 3 * 0.044715 * x * x)


def silu(x): return x * sigmoid(x)
def d_silu(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


print(gelu(1.0), d_gelu(1.0), silu(-2.0))
