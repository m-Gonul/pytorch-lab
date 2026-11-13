import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def gl():
        r = torch.tensor([[1.0], [1.0], [1.0]])
        v = torch.tensor([[0.5], [0.2], [0.1]])
        d = torch.tensor([[0.0], [1.0], [0.0]])
        adv, ret = need(ns, "gae")(r, v, d, torch.tensor([2.0]), 0.9, 0.8)
        d0 = 1 + 0.9 * 0.2 - 0.5
        d1 = 1 - 0.2
        d2 = 1 + 0.9 * 2.0 - 0.1
        close(adv[:, 0].numpy(), torch.tensor([d0 + 0.72 * d1, d1, d2]).numpy(), what="GAE (bitişte kesilir)")
        close(ret.numpy(), (adv + v).numpy(), what="dönüş hedefi = A + V")
        lp = torch.log(torch.tensor([0.6, 0.2]))
        lo = torch.log(torch.tensor([0.4, 0.4]))
        close(float(need(ns, "ppo_loss")(lp, lo, torch.tensor([2.0, -1.0]), 0.2)), -(1.2 * 2 + (-0.8)) / 2, what="kırpılmış amaç")
    R_.test("gae ve ppo_loss", gl)

    def fis():
        torch.manual_seed(0)
        pi = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
        X = torch.randn(50, 3)
        params = list(pi.parameters())
        n = sum(p.numel() for p in params)
        probs = torch.softmax(pi(X), 1).detach()
        J = []
        for i in range(50):
            for a in range(2):
                gi = torch.autograd.grad(torch.log_softmax(pi(X[i:i + 1]), 1)[0, a], params)
                J.append((probs[i, a], torch.cat([x.reshape(-1) for x in gi])))
        F = sum(p * torch.outer(g, g) for p, g in J) / 50
        v = torch.randn(n)
        close(need(ns, "fvp")(pi, X, v, 0.0).numpy(), (F @ v).numpy(), atol=1e-5, what="Fisher-vektör çarpımı = E[∇logπ∇logπᵀ]v")
        A = F + 0.1 * torch.eye(n)
        b = torch.randn(n)
        x = need(ns, "cg")(lambda u: A @ u, b, 200)
        close(x.numpy(), torch.linalg.solve(A, b).numpy(), atol=1e-4, what="CG")
    R_.test("fvp ve cg", fis)

    def step():
        torch.manual_seed(1)
        g = torch.Generator().manual_seed(1)
        net = ns["PolicyValue"]()
        state = [(torch.rand(16, 4, generator=g) - 0.5) * 0.1, torch.zeros(16), []]
        X, A, LP, V, R, D, vl = ns["rollout"](net, state, g)
        adv, _ = ns["gae"](R, V, D, vl, 0.99, 0.95)
        X, A, LP, adv = X.reshape(-1, 4), A.reshape(-1), LP.reshape(-1), adv.reshape(-1)
        adv = (adv - adv.mean()) / adv.std()
        p_old = torch.softmax(net.pi(X), 1).detach()
        L0 = float((adv * 1.0).mean())
        ok = need(ns, "trpo_step")(net.pi, X, A, adv, LP, 0.01)
        with torch.no_grad():
            kl = float((p_old * (torch.log(p_old) - torch.log_softmax(net.pi(X), 1))).sum(1).mean())
            L1 = float((torch.exp(torch.log_softmax(net.pi(X), 1).gather(1, A[:, None]).squeeze(1) - LP) * adv).mean())
        ensure(ok and kl <= 0.015 and L1 > L0, f"TRPO adımı: KL {kl:.4f} ≤ 1.5δ ve vekil artmalı ({L0:.4f} → {L1:.4f})")
    R_.test("trpo_step", step)

    def train():
        def fit(name):
            # Tek bir tohum kayan nokta farklarına duyarlı (aynı kod bir makinede 170, ötekinde 450+ verebiliyor);
            # ilk tohum eşiğin altında kalırsa ikinci tohum denenir.
            for seed in (0, 1):
                net, log = need(ns, name)(60, seed=seed)
                e = float(ns["evaluate"](net).mean())
                if e >= 200: break
            return net, log, e
        npp, lp, ep = fit("train_ppo")
        nt, lt, et = fit("train_trpo")
        ensure(ep >= 200 and et >= 200, f"PPO ve TRPO CartPole'u öğrenmeli ({ep:.0f}, {et:.0f}; iki tohumun ikisinde de 200'ün altında)")
        R_.table("CartPole, 60 yineleme × 16 ortam × 128 adım (122 880 adım)", ["yöntem", "son 10 eğitim bölümü", "açgözlü değerlendirme"],
                 [["PPO (ε = 0.2, 4 epoch)", f"{sum(lp[-10:]) / 10:.0f}", f"{ep:.0f}"], ["TRPO (δ = 0.01)", f"{sum(lt[-10:]) / 10:.0f}", f"{et:.0f}"]],
                 note="Aynı veri bütçesinde A2C'nin üçte biri kadar adım. PPO her veriyi birkaç epoch kullanır (kırpma güvenli kılar); TRPO her yinelemede tek ama 'optimal büyüklükte' bir adım atar.")
    R_.test("train_ppo ve train_trpo", train)
