def erode(A, B):
    ...


def dilate(A, B):
    ...


def hit_or_miss(I, B):
    ...


def boundary(A):
    ...


A = [[0] * 9 for _ in range(9)]
for y in range(2, 7):
    for x in range(2, 7):
        A[y][x] = 1
print(sum(map(sum, erode(A, [[1] * 3] * 3))))
