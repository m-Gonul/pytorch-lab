import math
import torch
from torch.distributions import Normal
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def ref(encode, decode, x, eps):
    mu, lv = encode(x)
    q = Normal(mu, torch.exp(0.5 * lv))
    vals = []
    for k in range(eps.shape[0]):
        z = mu + torch.exp(0.5 * lv) * eps[k]
        lw = Normal(decode(z), 1.0).log_prob(x).sum(-1) + Normal(0.0, 1.0).log_prob(z).sum(-1) - q.log_prob(z).sum(-1)
        vals.append(lw)
    lw = torch.stack(vals)
    return (torch.logsumexp(lw, 0) - math.log(len(vals))).mean()


def check(ns, R):
    src = ns["__source__"]
    R.test("iwae_bound döngüsüz", lambda: max_loops(src, 0, func="iwae_bound"))
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 6)).double()
    dec = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 6)).double()
    encode = lambda x: enc(x).chunk(2, -1)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 6, generator=g, dtype=torch.float64)

    def eq():
        eps = torch.randn(7, 10, 3, generator=g, dtype=torch.float64)
        close(need(ns, "iwae_bound")(encode, dec, x, eps), ref(encode, dec, x, eps))
    R.test("döngülü başvuru (torch.distributions) ile aynı, K = 7", eq)

    def k1():
        eps = torch.randn(1, 10, 3, generator=g, dtype=torch.float64)
        mu, lv = encode(x)
        z = mu + torch.exp(0.5 * lv) * eps[0]
        rec = Normal(dec(z), 1.0).log_prob(x).sum(-1)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)
        lw = rec + Normal(0.0, 1.0).log_prob(z).sum(-1) - Normal(mu, torch.exp(0.5 * lv)).log_prob(z).sum(-1)
        close(need(ns, "iwae_bound")(encode, dec, x, eps), lw.mean())
    R.test("K = 1 → tek örnekli ELBO", k1)

    def mono():
        vals = []
        with torch.no_grad():
            for K in (1, 5, 50, 500):
                vals.append(sum(ns["iwae_bound"](encode, dec, x, torch.randn(K, 10, 3, generator=g, dtype=torch.float64)).item() for _ in range(20)) / 20)
        need(ns, "iwae_bound")
        ensure(all(a < b for a, b in zip(vals, vals[1:])), f"K ile artmalı: {[round(v, 4) for v in vals]}")
        return "K = 1, 5, 50, 500 → " + ", ".join(f"{v:.4f}" for v in vals)
    R.test("sınır K ile sıkılaşıyor", mono)

    def grad():
        eps = torch.randn(4, 10, 3, generator=g, dtype=torch.float64)
        need(ns, "iwae_bound")(encode, dec, x, eps).backward()
        ensure(all(p.grad is not None for p in enc.parameters()), "kodlayıcı gradyan almalı")
    R.test("geri geçiş", grad)
    if "iwae_bound" not in ns:
        return
    enc32 = enc.float(); dec32 = dec.float()
    encode32 = lambda x: enc32(x).chunk(2, -1)
    xb = torch.randn(256, 6); eb = torch.randn(200, 256, 3)
    with torch.no_grad():
        t_loop = bench(lambda: ref(encode32, dec32, xb, eb), repeat=3)
        t_vec = bench(lambda: ns["iwae_bound"](encode32, dec32, xb, eb), repeat=3)
    R.table("K = 200, B = 256", ["yöntem", "süre"],
            [["K üzerinde döngü", fmt_time(t_loop)], ["senin (K·B tek batch)", fmt_time(t_vec)]],
            note="K örneği batch boyutuna katlamak: decoder bir kez, büyük bir matmul ile çalışır.")
