import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def one(X, Gs, s, act):
    H, o = X, []
    for G in Gs:
        H = act(s / X.shape[1] ** 0.5 * (H @ G.T)); o.append(H.std())
    return torch.stack(o)


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız katman döngüsü", lambda: max_loops(src, 1, func="std_vs_scale", why="ölçekler üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = torch.randn(128, 64, generator=g, dtype=torch.float64)
    Gs = torch.randn(30, 64, 64, generator=g, dtype=torch.float64)
    scales = torch.linspace(0.6, 2.4, 19, dtype=torch.float64)

    def eq():
        out = need(ns, "std_vs_scale")(X, Gs, scales, torch.relu)
        ensure(tuple(out.shape) == (19, 30), f"shape {tuple(out.shape)}")
        for i in (0, 9, 18):
            close(out[i], one(X, Gs, float(scales[i]), torch.relu), rtol=1e-9, what=f"s = {float(scales[i]):.2f}")
        ratio = out[:, -1] / X.std()
        k = int((ratio.log().abs()).argmin())
        return f"30 katmandan sonra std'yi en iyi koruyan ölçek s = {float(scales[k]):.2f} (teori √2 ≈ 1.41)"
    R.test("tek tek simülasyonla aynı; kritik ölçek ≈ √2", eq)
    if "std_vs_scale" not in ns:
        return
    Xf, Gf = X.float(), Gs.float()
    sc = torch.linspace(0.5, 2.5, 64)
    t_loop = bench(lambda: [one(Xf, Gf, float(s), torch.relu) for s in sc], repeat=3, number=1)
    t_vec = bench(lambda: ns["std_vs_scale"](Xf, Gf, sc, torch.relu), repeat=3, number=1)
    R.table("64 ölçek × 30 katman", ["yöntem", "süre", "döngüye göre"],
            [["for s: simülasyon", fmt_time(t_loop), "1×"], ["senin std_vs_scale", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
