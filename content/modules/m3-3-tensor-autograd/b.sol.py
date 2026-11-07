def broadcast_shape(s1, s2):
    n = max(len(s1), len(s2))
    a = (1,) * (n - len(s1)) + tuple(s1)
    b = (1,) * (n - len(s2)) + tuple(s2)
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"broadcast edilemez: {s1} ve {s2}")
    return tuple(out)


def unbroadcast2d(G, shape):
    n, m = len(G), len(G[0])
    if shape == ():
        return sum(sum(r) for r in G)
    if shape == (n, m):
        return [r[:] for r in G]
    if shape in ((m,), (1, m)):
        col = [sum(G[i][j] for i in range(n)) for j in range(m)]
        return col if shape == (m,) else [col]
    if shape == (n, 1):
        return [[sum(r)] for r in G]
    raise ValueError(f"desteklenmeyen shape {shape}")


print(broadcast_shape((2, 3), (3,)), broadcast_shape((4, 1), (1, 5)))   # (2, 3) (4, 5)
G = [[1, 2, 3], [4, 5, 6]]
print(unbroadcast2d(G, (3,)), unbroadcast2d(G, (2, 1)), unbroadcast2d(G, ()))   # [5, 7, 9] [[6], [15]] 21
