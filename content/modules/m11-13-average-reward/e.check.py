from labcheck import close, need, ensure, max_loops
import torch


def check(ns, R_):
    R_.test("ortam döngüsü yok", lambda: max_loops(ns["__source__"], 1, func="batched_access"))
    def t():
        out = need(ns, "batched_access")(n_envs=16, steps=3000, alpha=.02, beta=.005, seed=3)
        ensure(tuple(out["Q"].shape)==(16,44,2), "ayrı Q tabloları")
        close(out["visits"].sum(dim=1).numpy(), [3000]*16, what="ortam başına3000gerçek geçiş")
        close(out["rho"].numpy(), (.25*out["Q"].sum(dim=(1,2))).numpy(), atol=1e-9, what="her ortamda ayrı gain değişmezi")
        close(out["Q"][:, :4, 1].numpy(), torch.zeros(16,4).numpy(), what="geçersiz eylemler")
        ensure(bool(torch.isfinite(out["Q"]).all()) and bool((out["reward_mean"]>=0).all()) and bool((out["reward_mean"]<=8).all()), "sonlu sonuçlar")
        ensure(float(out["reward_mean"].std())>0, "bağımsız ortamlar farklı sonuç üretiyor")
        R_.table("16ajan ×3000geçiş", ["ortalama gerçek ödül", "koşular arası std"], [[float(out["reward_mean"].mean()),float(out["reward_mean"].std())]])
    R_.test("Bağımsız vektörleştirilmiş kontrol", t)
