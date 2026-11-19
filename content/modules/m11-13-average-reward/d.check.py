from labcheck import close, need, ensure
import torch


def check(ns, R_):
    def t():
        Q = torch.zeros(44, 2, dtype=torch.float64, requires_grad=True)
        Q.data[1, 0] = 3.0
        updated, rho = need(ns, "differential_torch_step")(Q, .5, 0, 1, 2., 1, 0, .1, .02)
        # target2-.5+3=4.5; only the current prediction receives a gradient.
        close(updated[0, 1].detach().numpy(), .45, what="doğru current-only adım")
        close(updated[1, 0].detach().numpy(), 3., what="hedef parametresi değişmedi")
        close(rho, .59, what="aynı δ ile gain adımı")
        ensure(updated.is_leaf and updated.requires_grad, "yeni adım için temiz yaprak")
        out = need(ns, "torch_sarsa")(steps=3000, alpha=.02, beta=.005, seed=3)
        close(sum(out["visits"]), 3000, what="fiziksel geçişler")
        close(out["rho"], .25 * float(out["Q"].sum()), atol=1e-9, what="gain / toplam ağırlık değişmezi")
        close(out["Q"][:4, 1].numpy(), [0] * 4, what="boş0geçersiz kabul")
        ensure(bool(torch.isfinite(out["Q"]).all()) and 0<=out["reward_mean"]<=8, "sonlu değerler")
        R_.table("3000 geçiş / autograd", ["gerçek ortalama", "rho"], [[out["reward_mean"], out["rho"]]])
    R_.test("Detach desteği ve gerçek diferansiyel kontrol", t)
