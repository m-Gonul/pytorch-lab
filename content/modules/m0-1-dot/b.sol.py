import math


def dot(u, v):
    if len(u) != len(v):
        raise ValueError(f"uzunluklar farklı: {len(u)} ≠ {len(v)}")
    s = 0
    for a, b in zip(u, v):
        s += a * b
    return s


def norm(u):
    return math.sqrt(dot(u, u))


def cosine(u, v):
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("sıfır vektörün yönü yok; kosinüs tanımsız")
    return dot(u, v) / (nu * nv)


print(dot([1, 2, 3], [4, -5, 6]))      # 12
print(norm([1, 2, 3]))                 # 3.7416...
print(cosine([1, 2, 3], [4, -5, 6]))   # 0.3654...
