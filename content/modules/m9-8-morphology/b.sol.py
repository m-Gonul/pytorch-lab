def _get(A, y, x):
    return A[y][x] if 0 <= y < len(A) and 0 <= x < len(A[0]) else 0


def _offsets(B):
    r = len(B) // 2
    return [(i - r, j - r) for i in range(len(B)) for j in range(len(B)) if B[i][j] == 1]


def erode(A, B):
    H, W = len(A), len(A[0])
    off = _offsets(B)
    return [[int(all(_get(A, y + dy, x + dx) for dy, dx in off)) for x in range(W)] for y in range(H)]


def dilate(A, B):
    H, W = len(A), len(A[0])
    off = _offsets(B)
    return [[int(any(_get(A, y - dy, x - dx) for dy, dx in off)) for x in range(W)] for y in range(H)]


def hit_or_miss(I, B):
    H, W = len(I), len(I[0])
    r = len(B) // 2
    cells = [(i - r, j - r, B[i][j]) for i in range(len(B)) for j in range(len(B)) if B[i][j] != -1]
    return [[int(all(_get(I, y + dy, x + dx) == v for dy, dx, v in cells)) for x in range(W)] for y in range(H)]


def boundary(A):
    E = erode(A, [[1] * 3] * 3)
    return [[a - e for a, e in zip(ra, re)] for ra, re in zip(A, E)]


A = [[0] * 9 for _ in range(9)]
for y in range(2, 7):
    for x in range(2, 7):
        A[y][x] = 1
print(sum(map(sum, erode(A, [[1] * 3] * 3))))
