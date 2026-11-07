import cmath


def dft(x):
    N = len(x)
    return [sum(x[n] * cmath.exp(-2j * cmath.pi * k * n / N) for n in range(N)) for k in range(N)]


def fft(x):
    N = len(x)
    if N <= 1:
        return [complex(v) for v in x]
    if N & (N - 1):
        return dft(x)
    E, O = fft(x[0::2]), fft(x[1::2])
    out = [0j] * N
    for k in range(N // 2):
        t = cmath.exp(-2j * cmath.pi * k / N) * O[k]
        out[k] = E[k] + t
        out[k + N // 2] = E[k] - t
    return out


def idft(X):
    N = len(X)
    return [v.conjugate() / N for v in fft([c.conjugate() for c in X])]


def dft2(img):
    rows = [fft(r) for r in img]
    cols = [fft([rows[i][j] for i in range(len(rows))]) for j in range(len(rows[0]))]
    return [[cols[j][i] for j in range(len(cols))] for i in range(len(rows))]


print([round(abs(v), 3) for v in fft([1, 2, 3, 4])])
