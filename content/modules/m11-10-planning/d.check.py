import torch
from labcheck import close, need, ensure


class TrueModel(torch.nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, x, a):
        return self.step(x, a)[0]


def check(ns, R_):
    g = torch.Generator().manual_seed(0)

    def data():
        X, A, X2 = need(ns, "random_data")(200, g)
        ensure(X.shape[1] == 4 and X.shape[0] == A.shape[0] == X2.shape[0] > 3000, "veri şekilleri")
        close(X2.numpy(), ns["cartpole_step"](X, A)[0].numpy(), atol=1e-6, what="geçişler gerçek fizikten")
        m = need(ns, "Model")()
        loss = need(ns, "fit")(m, X, A, X2, 100, 0)
        Xt, At, X2t = ns["random_data"](20, torch.Generator().manual_seed(9))
        with torch.no_grad():
            err = ((m(Xt, At) - X2t) / (X2t - Xt).std(0)).pow(2).mean()
            base = ((Xt - X2t) / (X2t - Xt).std(0)).pow(2).mean()
        ensure(float(err) < 0.01 * float(base), f"model fark tahmininde iyi olmalı ({float(err):.2e} vs {float(base):.2e})")
        lens_true = [need(ns, "run_mpc")(TrueModel(ns["cartpole_step"]), g, 200) for _ in range(2)]
        lens = [ns["run_mpc"](m, g, 200) for _ in range(3)]
        ensure(min(lens) >= 150, f"öğrenilmiş modelle MPC dengede tutmalı ({lens})")
        R_.table("CartPole, rastgele atış MPC (500 dizi × 20 adım)", ["model", "bölüm uzunlukları (en fazla 200)"],
                 [["gerçek fizik", ", ".join(map(str, lens_true))], ["öğrenilmiş ağ (yalnız rastgele veri, ~4000 geçiş)", ", ".join(map(str, lens))]],
                 note=f"Rastgele politika ~22 adım dayanır. Model test hatası (normalize Δ, MSE) {float(err):.1e}. Hiç ödül sinyali olmadan toplanan veriden öğrenilen model, bilinen bir maliyetle planlanınca görevi çözer — modelsiz DQN/A2C'nin yüz binlerce adımına karşı birkaç bin geçiş.")
    R_.test("model öğrenme ve MPC", data)
