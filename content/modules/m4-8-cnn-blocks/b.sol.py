def count(layers, in_shape):
    C, H, W = in_shape
    flat = None
    params = macs = 0
    for L in layers:
        t = L["type"]
        if t == "conv":
            g = L.get("groups", 1)
            p = L["k"] ** 2 * (L["cin"] // g) * L["cout"]
            params += p + (L["cout"] if L.get("bias", True) else 0)
            macs += H * W * p
            C = L["cout"]
        elif t == "pool":
            H, W = H // L["k"], W // L["k"]
        elif t == "bn":
            params += 2 * L["c"]
        elif t == "flatten":
            flat = C * H * W
        elif t == "linear":
            params += L["in"] * L["out"] + L["out"]
            macs += L["in"] * L["out"]
            flat = L["out"]
    return params, macs, ((flat,) if flat is not None else (C, H, W))


def inception_params(cin, c1, c3r, c3, c5r, c5, pp):
    p = cin * c1 + cin * c3r + 9 * c3r * c3 + cin * c5r + 25 * c5r * c5 + cin * pp
    return p, c1 + c3 + c5 + pp


def lrn(a, k=2, n=5, alpha=1e-4, beta=0.75):
    C = len(a)
    out = []
    for i in range(C):
        s = sum(a[j] ** 2 for j in range(max(0, i - n // 2), min(C, i + n // 2 + 1)))
        out.append(a[i] / (k + alpha * s) ** beta)
    return out


cnn2 = [{"type": "conv", "k": 7, "cin": 1, "cout": 16, "groups": 1, "bias": True}, {"type": "pool", "k": 2}]
print(count(cnn2, (1, 28, 28)))
