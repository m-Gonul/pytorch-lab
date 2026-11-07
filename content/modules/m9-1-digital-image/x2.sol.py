def distance_transform(bin_img, metric):
    H, W = len(bin_img), len(bin_img[0])
    INF = H + W + 10
    d = [[0 if bin_img[y][x] == 0 else INF for x in range(W)] for y in range(H)]
    fwd = [(-1, 0), (0, -1)] + ([(-1, -1), (-1, 1)] if metric == "d8" else [])
    bwd = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if metric == "d8" else [])
    for y in range(H):
        for x in range(W):
            for dy, dx in fwd:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    d[y][x] = min(d[y][x], d[yy][xx] + 1)
    for y in range(H - 1, -1, -1):
        for x in range(W - 1, -1, -1):
            for dy, dx in bwd:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    d[y][x] = min(d[y][x], d[yy][xx] + 1)
    return d


B = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 0, 1, 1],
     [1, 1, 1, 1, 1]]
for row in distance_transform(B, "d4"):
    print(row)
