import time
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("discounted_returns_vec döngüsüz", lambda: max_loops(src, 0, func="discounted_returns_vec"))
    R.test("rollout: tek döngü (zaman)", lambda: max_loops(src, 1, func="rollout"))
    Chain, make_policy = need(ns, "Chain", "sınıf"), need(ns, "make_policy")

    def dr():
        g = torch.Generator().manual_seed(0)
        r = (torch.rand(5, 12, generator=g, dtype=torch.float64) < 0.2).double()
        want = torch.zeros_like(r)
        for b in range(5):
            G = 0.0
            for t in reversed(range(12)):
                G = r[b, t] + 0.9 * G
                want[b, t] = G
        close(need(ns, "discounted_returns_vec")(r, 0.9), want)
    R.test("discounted_returns_vec = geriye döngü", dr)

    def ro():
        torch.manual_seed(0)
        env = Chain(B=16)
        logp, r, mask = need(ns, "rollout")(make_policy(), env, 30)
        ensure(tuple(logp.shape) == tuple(r.shape) == tuple(mask.shape) == (16, 30), "şekiller (B, T)")
        ensure(torch.all(mask[:, 0] == 1) and torch.all(mask[:, 1:] <= mask[:, :-1]), "mask başta 1, sonra artmayan")
        ensure(torch.all(r * (1 - mask) == 0), "bitişten sonra ödül olmamalı")
        ensure(torch.all(r.sum(1) <= 1), "her bölümde en fazla bir +1")
        ensure(logp.requires_grad, "logp gradyanlı olmalı")
    R.test("rollout: maske ve ödüller", ro)

    def tr():
        torch.manual_seed(0)
        pi = make_policy()
        t0 = time.perf_counter()
        need(ns, "train_batched")(pi, Chain(B=64))
        dt = time.perf_counter() - t0
        env = Chain(); obs = env.reset(); steps = 0
        with torch.no_grad():
            while True:
                obs, r, done = env.step(pi(obs).argmax(-1)); steps += 1
                if done.item():
                    break
        ensure(env.s.item() == 7 and steps == 4, f"açgözlü politika {steps} adımda {env.s.item()}. hücrede bitti")
        return f"60 iterasyon × 64 bölüm = 3840 bölüm, {fmt_time(dt)}"
    R.test("batch'li REINFORCE zinciri çözüyor", tr)
    if "rollout" not in ns:
        return
    torch.manual_seed(0)
    pi = make_policy()
    env1, env64 = Chain(B=1), Chain(B=256)

    def seq():
        for _ in range(256):
            ns["rollout"](pi, env1, 30)
    t_seq = bench(seq, repeat=1, number=1)
    t_bat = bench(lambda: ns["rollout"](pi, env64, 30), repeat=3)
    R.table("256 bölüm × 30 adım", ["yöntem", "süre"], [["ortamlar tek tek", fmt_time(t_seq)], ["256 ortam paralel", fmt_time(t_bat)]],
            note="Gerçek RL kütüphaneleri (ör. vektörize gym ortamları, GPU simülatörleri) bu yüzden ortamları batch'ler.")
