import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: (max_loops(src, 0, func="mixture_eps"), max_loops(src, 0, func="class_logprob_grad"), max_loops(src, 1, func="ddim")))
    MU, LAB, SIG = ns["MU"], ns["LAB"], ns["SIG"]
    wu = np.full(8, 1 / 8)

    def logp(x, abar, w):
        v = abar * SIG ** 2 + 1 - abar
        d2 = ((x[:, None] - np.sqrt(abar) * MU[None]) ** 2).sum(-1)
        return np.log((w[None] * np.exp(-d2 / (2 * v))).sum(1))

    def t_exact():
        rng = np.random.default_rng(0)
        x = rng.normal(0, 2, (5, 2))
        e = need(ns, "mixture_eps")(x, 0.6, wu)
        h = 1e-6
        g = np.stack([(logp(x + h * d, 0.6, wu) - logp(x, 0.6, wu)) / h for d in np.eye(2)], 1)
        close(e, -np.sqrt(0.4) * g, atol=1e-4, what="ε = −σ·∇log p(x_t)")
        wc = (LAB == 1) / 2.0
        lp_y = lambda z: logp(z, 0.6, wc) + np.log(0.25) - logp(z, 0.6, wu)
        gy = np.stack([(lp_y(x + h * d) - lp_y(x)) / h for d in np.eye(2)], 1)
        close(need(ns, "class_logprob_grad")(x, 0.6, 1), gy, atol=1e-4, what="∇log p(y|x_t) (Bayes)")
        for s in (0.5, 3.0):
            cg = ns["mixture_eps"](x, 0.6, wu) - s * np.sqrt(0.4) * ns["class_logprob_grad"](x, 0.6, 1)
            cf = ns["mixture_eps"](x, 0.6, wu) + s * (ns["mixture_eps"](x, 0.6, wc) - ns["mixture_eps"](x, 0.6, wu))
            close(cg, cf, atol=1e-10, what=f"sınıflandırıcı yönlendirmesi = CFG (s = {s})")
    R_.test("kesin skorlar", t_exact)

    def t_sample():
        betas = np.linspace(1e-4, 0.02, 1000)
        ab = np.cumprod(1 - betas)[::5]
        xT = np.random.default_rng(1).normal(size=(3000, 2))
        x0 = need(ns, "ddim")(xT, ab, lambda x, a: ns["mixture_eps"](x, a, wu))
        near = np.argmin(((x0[:, None] - MU[None]) ** 2).sum(-1), 1)
        cnt = np.bincount(near, minlength=8) / len(x0)
        ensure(np.all(np.abs(cnt - 1 / 8) < 0.03), "koşulsuz: 8 mod eşit")
        wc = (LAB == 1) / 2.0
        rows, R = [], {}
        for s in (0.0, 1.0, 2.0, 4.0, 8.0):
            f = lambda x, a: ns["mixture_eps"](x, a, wu) + s * (ns["mixture_eps"](x, a, wc) - ns["mixture_eps"](x, a, wu))
            x0 = ns["ddim"](xT, ab, f)
            d2 = ((x0[:, None] - MU[None]) ** 2).sum(-1)
            r = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * SIG ** 2))
            r /= r.sum(1, keepdims=True)
            py = r[:, LAB == 1].sum(1).mean()
            k = np.argmin(d2, 1)
            two = np.bincount(k[LAB[k] == 1], minlength=8)[LAB == 1]
            R[s] = (py, np.linalg.norm(x0, axis=1).mean(), np.sqrt(d2.min(1)).mean())
            rows.append([s, f"{py:.3f}", f"{two[0]}/{two[1]}", f"{R[s][1]:.2f}", f"{R[s][2]:.2f}"])
        ensure(R[0.0][0] < 0.35 and R[1.0][0] > 0.98, "s = 0 koşulsuz, s = 1 koşullu")
        ensure(R[8.0][1] > R[1.0][1] + 0.5, "büyük s: örnekler dışa taşar")
        R_.table("Kesin ε ile yönlendirilmiş DDIM (200 adım), hedef sınıf 1 (açı 45° ve 225°)", ["s", "ort. p(y = 1 | x₀)", "iki modun payı", "ort. yarıçap (veri: 4)", "en yakın merkeze uzaklık (veri ≈ 0.63)"], rows,
                 note="Model kusursuz olduğu hâlde s > 1 dağılımı değiştirir: p(x)·p(y|x)^s. Sınıf zaten kesin olduğundan kesinlik doymuştur; bunun yerine örnekler diğer sınıflardan 'kaçmak' için dışarı itilir (yarıçap 4'ün üstüne) — görüntülerdeki aşırı doymuş renkler ve abartılı özelliklerin oyuncak karşılığı. Her iki mod korunur: yönlendirme sınıf içi modları değil, sınıflar arası ayrımı keskinleştirir.")
    R_.test("yönlendirilmiş örnekleme", t_sample)
