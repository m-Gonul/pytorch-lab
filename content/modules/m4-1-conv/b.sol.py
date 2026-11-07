def pad2d(x, p):
    out = []
    for ch in x:
        W = len(ch[0]) + 2 * p
        rows = [[0] * W for _ in range(p)]
        rows += [[0] * p + list(r) + [0] * p for r in ch]
        rows += [[0] * W for _ in range(p)]
        out.append(rows)
    return out


def conv2d_loops(x, w, b, stride=1, pad=0):
    xp = pad2d(x, pad)
    C, H, W = len(xp), len(xp[0]), len(xp[0][0])
    F, KH, KW = len(w), len(w[0][0]), len(w[0][0][0])
    OH, OW = (H - KH) // stride + 1, (W - KW) // stride + 1
    out = []
    for f in range(F):
        plane = []
        for i in range(OH):
            row = []
            for j in range(OW):
                s = b[f]
                for c in range(C):
                    for a in range(KH):
                        for bb in range(KW):
                            s += w[f][c][a][bb] * xp[c][i * stride + a][j * stride + bb]
                row.append(s)
            plane.append(row)
        out.append(plane)
    return out


x = [[[1, 2, 0], [0, 1, 3], [2, 1, 0]]]
w = [[[[1, 0], [-1, 2]]]]
print(conv2d_loops(x, w, [0]))      # [[[3, 7], [0, 0]]]
