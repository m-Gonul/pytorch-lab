import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def np_():
        mu, var = need(ns, "normal_posterior")(torch.tensor([3.0, 0.0]), torch.tensor([3.0, 0.0]), 0.0, 1.0, 1.0)
        close(mu.numpy(), torch.tensor([0.75, 0.0]).numpy(), what="sonsal ortalama")
        close(var.numpy(), torch.tensor([0.25, 1.0]).numpy(), what="sonsal varyans")
        g = torch.Generator().manual_seed(0)
        x = 2.0 + 0.5 * torch.randn(10000, generator=g, dtype=torch.float64)
        mu, var = ns["normal_posterior"](x.sum(), 10000, 0.0, 100.0, 0.25)
        ensure(abs(float(mu) - 2.0) < 0.02 and float(var) < 1e-4, "çok veride MLE'ye yakınsar")
    R_.test("normal_posterior", np_)

    def k():
        probs = [0.9, 0.8, 0.7, 0.5, 0.3]
        rows, res = [], {}
        for kk in (1, 10, 100):
            r = need(ns, "thompson")(probs, 1000, 200, kk, 3)
            ensure(r.shape == (1000,) and bool((r[1:] >= r[:-1] - 1e-12).all()), "artan kümülatif pişmanlık")
            res[kk] = r
            rows.append([kk, f"{float(r[99]):.1f}", f"{float(r[-1]):.1f}"])
        ensure(float(res[1][-1]) < float(res[100][-1]), "k = 1 daha iyi keşfetmeli")
        rg = need(ns, "gaussian_thompson")([1.0, 0.5, 0.0, -0.5], 1000, 200, 1.0, 3)
        rows.append(["Normal–Normal (k = 1, kollar 1, 0.5, 0, −0.5; σ = 1)", f"{float(rg[99]):.1f}", f"{float(rg[-1]):.1f}"])
        ensure(float(rg[-1]) < 100, "Gauss Thompson öğrenmeli")
        R_.table("Thompson örneklemesi, 200 deney", ["k", "t = 100", "t = 1000"], rows,
                 note="k büyüdükçe R̂ sonsal ortalamaya yaklaşır: rastgelelik — ve keşif — kaybolur; bazı deneyler alt-optimal kola takılır ve pişmanlık uzun vadede artar.")
    R_.test("k'nın etkisi ve Gauss Thompson", k)
