import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    def mk(seed=0, **kw):
        torch.manual_seed(seed)
        cfg = dict(vocab=29, block_size=40, d=32, n_layer=2, n_head=4); cfg.update(kw)
        return need(ns, "GPT", "sınıf")(**cfg), cfg

    def count():
        m, c = mk()
        V, T, d, L = c["vocab"], c["block_size"], c["d"], c["n_layer"]
        got = sum(p.numel() for p in m.parameters())
        want = V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d
        ensure(got == want, f"{got} parametre; beklenen {want} (12d²+13d blok başına, tied head)")
        return f"{got:,} parametre"
    R.test("parametre sayısı formülle aynı", count)

    def tie():
        m, _ = mk()
        ensure(isinstance(getattr(m, "head", None), nn.Linear) and m.head.weight is m.wte.weight, "head.weight, wte.weight ile aynı nesne olmalı")
    R.test("weight tying", tie)

    def shapes():
        m, _ = mk()
        idx = torch.randint(0, 29, (3, 11))
        logits, loss = m(idx)
        ensure(tuple(logits.shape) == (3, 11, 29) and loss is None, "targets yokken (logits (B,T,V), None)")
        _, loss = m(idx, idx)
        ensure(loss.dim() == 0, "loss skaler olmalı")
    R.test("çıktı şekilleri", shapes)

    def causal():
        m, _ = mk(1)
        m.eval()
        idx = torch.randint(0, 29, (2, 20))
        idx2 = idx.clone(); idx2[:, 12:] = (idx2[:, 12:] + 3) % 29
        with torch.no_grad():
            a, _ = m(idx); b, _ = m(idx2)
        close(a[:, :12], b[:, :12], atol=1e-5, what="ilk 12 konum")
        ensure(not torch.allclose(a[:, 12:], b[:, 12:]), "sonraki konumlar değişmeliydi")
    R.test("nedensellik", causal)

    def overfit():
        text = "pytorch ile gpt yaziyoruz; once elle, sonra vektorel."
        chars = sorted(set(text))
        ids = torch.tensor([chars.index(c) for c in text])[None]
        m, _ = mk(2, vocab=len(chars), block_size=len(text))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for _ in range(250):
            _, loss = m(ids[:, :-1], ids[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
        ensure(loss.item() < 0.1, f"250 adım sonra kayıp {loss.item():.3f}; < 0.1 bekleniyordu")
        m.eval()
        out = ids[:, :1]
        with torch.no_grad():
            for _ in range(len(text) - 1):
                out = torch.cat([out, m(out)[0][:, -1].argmax(-1, keepdim=True)], 1)
        s = "".join(chars[i] for i in out[0].tolist())
        return f"kayıp {loss.item():.4f}; greedy: “{s}”"
    R.test("kısa metni ezberliyor", overfit)
