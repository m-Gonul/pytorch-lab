def sad(a, b):
    return sum(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _block(img, y, x, bs):
    return [row[x:x + bs] for row in img[y:y + bs]]


def block_match(prev, cur, bs, R):
    H, W = len(cur), len(cur[0])
    out = []
    for by in range(0, H - bs + 1, bs):
        row = []
        for bx in range(0, W - bs + 1, bs):
            c = _block(cur, by, bx, bs)
            best = None
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    y, x = by + dy, bx + dx
                    if 0 <= y <= H - bs and 0 <= x <= W - bs:
                        s = sad(c, _block(prev, y, x, bs))
                        if best is None or s < best[0]:
                            best = (s, dy, dx)
            row.append((best[1], best[2]))
        out.append(row)
    return out


def compensate(prev, mvs, bs):
    out = [list(r) for r in prev]
    for i, row in enumerate(mvs):
        for j, (dy, dx) in enumerate(row):
            for y in range(bs):
                for x in range(bs):
                    out[i * bs + y][j * bs + x] = prev[i * bs + y + dy][j * bs + x + dx]
    return out


prev = [[(x * 7 + y * 3) % 50 for x in range(16)] for y in range(16)]
cur = [row[1:] + row[:1] for row in prev]
print(block_match(prev, cur, 8, 2))
