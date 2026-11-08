import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        MU, LAB = ns["MU"], ns["LAB"]
        wu = np.full(8, 1 / 8)
        rng = np.random.default_rng(0)
        k = rng.integers(0, 8, 500)
        x0 = MU[k] + 0.5 * rng.normal(size=(500, 2))
        rows = []
        errs = {}
        for steps in (10, 50, 200):
            ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))[::1000 // steps]
            f = lambda x, a: ns["mixture_eps"](x, a, wu)
            xT = need(ns, "ddim_invert")(x0, ab, f)
            rec = need(ns, "ddim")(xT, ab, f)
            errs[steps] = np.abs(rec - x0).mean()
            rows.append([steps, f"{errs[steps]:.4f}", f"{xT.std():.2f}"])
        ensure(errs[200] < 0.05 and errs[200] < errs[10], f"ters çevirme hatası {errs}")
        ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))[::5]
        f = lambda x, a: ns["mixture_eps"](x, a, wu)
        sel = LAB[k] == 0
        xT = ns["ddim_invert"](x0[sel], ab, f)
        wc = (LAB == 2) / 2.0
        edit = ns["ddim"](xT, ab, lambda x, a: ns["mixture_eps"](x, a, wu) + 4 * (ns["mixture_eps"](x, a, wc) - ns["mixture_eps"](x, a, wu)))
        kk = np.argmin(((edit[:, None] - MU[None]) ** 2).sum(-1), 1)
        frac = (LAB[kk] == 2).mean()
        ensure(frac > 0.9, f"düzenleme sınıf 2'ye taşır ({frac:.2f})")
        R_.table("DDIM ters çevirme: x₀ → x_T → x₀ (kesin ε)", ["adım", "ort. |x̂₀ − x₀|", "x_T std"], rows,
                 note=f"Deterministik DDIM bir ODE'nin ayrıklaştırmasıdır: tersine çalıştırınca görüntüyü gürültü uzayında bir 'koda' çevirir, adım sayısı arttıkça yeniden kurulum kusursuzlaşır. Sınıf 0'dan ters çevrilen örnekler, aynı x_T'den sınıf 2 yönlendirmesiyle (s = 4) çözülünce %{frac * 100:.0f}'i sınıf 2'ye geçer — gürültü kodu konum/stil bilgisini taşırken koşul anlamı değiştirir (görüntü düzenlemenin temeli: SDEdit, prompt-to-prompt, null-text inversion).")
    R_.test("ters çevirme ve düzenleme", t)
