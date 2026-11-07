def vibe_classify(samples, frame, R=20, nmin=2):
    H, W = len(frame), len(frame[0])
    out = [[1] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            v, c = frame[y][x], 0
            for s in samples:
                if abs(v - s[y][x]) < R:
                    c += 1
                    if c >= nmin:
                        out[y][x] = 0
                        break
    return out


def running_average(frames, alpha):
    B = [[float(v) for v in row] for row in frames[0]]
    out = [B]
    for f in frames[1:]:
        B = [[(1 - alpha) * b + alpha * v for b, v in zip(rb, rf)] for rb, rf in zip(B, f)]
        out.append(B)
    return out


def ema_masks(frames, alpha, T):
    Bs = running_average(frames, alpha)
    return [[[int(abs(v - b) > T) for v, b in zip(rf, rb)] for rf, rb in zip(f, B)] for f, B in zip(frames[1:], Bs[:-1])]


print(vibe_classify([[[100]], [[130]], [[95]], [[180]], [[140]]], [[110]]))
