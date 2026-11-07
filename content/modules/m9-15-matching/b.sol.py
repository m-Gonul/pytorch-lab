import math


def ncc(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else 0.0


def match_template(img, t):
    h, w = len(t), len(t[0])
    H, W = len(img), len(img[0])
    tf = [v for row in t for v in row]
    return [[ncc([img[y + i][x + j] for i in range(h) for j in range(w)], tf) for x in range(W - w + 1)] for y in range(H - h + 1)]


def solve_h4(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    M = [row + [r] for row, r in zip(A, b)]
    n = 8
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    h = [0.0] * n
    for r in range(n - 1, -1, -1):
        h[r] = (M[r][n] - sum(M[r][k] * h[k] for k in range(r + 1, n))) / M[r][r]
    return [h[0:3], h[3:6], [h[6], h[7], 1.0]]


def apply_h(H, x, y):
    X = H[0][0] * x + H[0][1] * y + H[0][2]
    Y = H[1][0] * x + H[1][1] * y + H[1][2]
    Z = H[2][0] * x + H[2][1] * y + H[2][2]
    return X / Z, Y / Z


print(ncc([1, 2, 3, 4], [2, 1, 4, 3]))
