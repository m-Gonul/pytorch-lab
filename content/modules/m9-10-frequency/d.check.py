import numpy as np
import torch
from skimage import data, color
from labcheck import close, need, ensure

N = 128


def tiles():
    out = []
    for im in [data.camera(), color.rgb2gray(data.astronaut()) * 255, color.rgb2gray(data.coffee()) * 255, data.moon()]:
        im = im.astype(np.float32) / 255
        for i in range(0, im.shape[0] - N + 1, N):
            for j in range(0, im.shape[1] - N + 1, N):
                out.append(im[i:i + N, j:j + N])
    return torch.tensor(np.stack(out))


def check(ns, R):
    def otf():
        H = need(ns, "gaussian_otf")(N, 2.0)
        ensure(tuple(H.shape) == (N, N // 2 + 1), "rfft2 düzeni")
        psf = np.zeros((N, N)); r = np.arange(N); r = np.minimum(r, N - r)
        psf = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / 8.0); psf /= psf.sum()
        close(H.double().numpy(), np.real(np.fft.rfft2(psf)), atol=1e-4, what="dairesel Gauss PSF'nin DFT'si")
    R.test("gaussian_otf", otf)

    def rest():
        H = ns["gaussian_otf"](N, 2.0).float()
        train = tiles()
        test = torch.tensor(data.coins()[:N, :N].astype(np.float32) / 255)
        g = torch.Generator().manual_seed(1)
        y = need(ns, "degrade")(test, H, 0.01, g)
        P = lambda a: 10 * torch.log10(1 / ((a - test) ** 2).mean()).item()
        inv = P(need(ns, "apply")(1 / H, y))
        wk = {K: P(ns["apply"](need(ns, "wiener")(H, K), y)) for K in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)}
        Kb = max(wk, key=wk.get)
        G = need(ns, "learn_filter")(train, H, 0.01)
        pl = P(ns["apply"](G, y))
        ensure(pl > wk[Kb] - 0.3, f"öğrenilen {pl:.2f} dB, en iyi Wiener {wk[Kb]:.2f} dB")
        ensure(pl > P(y) + 1.0, "bulanık girdiden en az 1 dB iyi olmalı")
        R.table("coins (eğitimde yok), Gauss σ = 2 bulanıklık + σ = 0.01 gürültü", ["yöntem", "PSNR (dB)"],
                [["bulanık + gürültülü", f"{P(y):.2f}"], ["ters süzgeç 1/H", f"{inv:.1f}"], [f"Wiener (en iyi K = {Kb:g})", f"{wk[Kb]:.2f}"], ["öğrenilen G (600 adım)", f"{pl:.2f}"]],
                note="Ters süzgeç, H ≈ e⁻²⁰⁰ olan frekanslarda gürültüyü astronomik ölçüde büyütür.")
        prof = G[0, :40].numpy(); wp = ns["wiener"](H, Kb)[0, :40].numpy()
        c = float(np.corrcoef(prof, wp)[0, 1])
        ensure(c > 0.9, f"öğrenilen kazanç profili Wiener'e benzememiş (korelasyon {c:.2f})")
    R.test("öğrenilen frekans süzgeci ≈ Wiener", rest)
