import torch
from torch import nn
from labcheck import need, ensure


def block_params(i, o, s):
    p = 9 * i * o + 2 * o + 9 * o * o + 2 * o
    if s != 1 or i != o:
        p += i * o + 2 * o
    return p


def check(ns, R):
    def blk():
        B = need(ns, "BasicBlock", "sınıf")
        a, b = B(16, 16), B(16, 32, 2)
        ensure(isinstance(a.shortcut, nn.Identity), "aynı shape'te shortcut nn.Identity olmalı")
        ensure(not isinstance(b.shortcut, nn.Identity), "stride/kanal değişince projeksiyon gerekli")
        ensure(sum(p.numel() for p in a.parameters()) == block_params(16, 16, 1), "BasicBlock(16, 16) parametre sayısı")
        ensure(sum(p.numel() for p in b.parameters()) == block_params(16, 32, 2), "BasicBlock(16, 32, 2) parametre sayısı")
        y = b(torch.randn(2, 16, 8, 8))
        ensure(tuple(y.shape) == (2, 32, 4, 4), f"çıktı shape {tuple(y.shape)}")
        ensure(bool((y >= 0).all()), "son ReLU eksik")
    R.test("BasicBlock: shortcut türleri, parametreler, shape", blk)

    def net():
        m = need(ns, "SmallResNet", "sınıf")(10)
        want = 3 * 9 * 16 + 32 + block_params(16, 16, 1) * 2 + block_params(16, 32, 2) + block_params(32, 32, 1) + block_params(32, 64, 2) + block_params(64, 64, 1) + 64 * 10 + 10
        got = sum(p.numel() for p in m.parameters())
        ensure(got == want, f"parametre sayısı {got}, {want} bekleniyordu")
        ensure(tuple(m(torch.randn(2, 3, 32, 32)).shape) == (2, 10), "çıktı shape (2, 10)")
        return f"{got:,} parametre"
    R.test("SmallResNet: parametre sayısı ve çıktı", net)

    def train():
        g = torch.Generator().manual_seed(0)
        n = 256
        y = torch.randint(0, 4, (n,), generator=g)
        X = torch.randn(n, 3, 16, 16, generator=g) * 0.5
        for k in range(4):
            r, c = (k // 2) * 8, (k % 2) * 8
            X[y == k, :, r:r + 8, c:c + 8] += 1.5
        torch.manual_seed(0)
        m = need(ns, "SmallResNet", "sınıf")(4)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(40):
            idx = torch.randint(0, n, (64,), generator=g)
            loss = nn.functional.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(X).argmax(1) == y).float().mean().item()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"sentetik 4 sınıf (parlak çeyrek), 40 adım: doğruluk {acc:.3f}"
    R.test("SmallResNet öğreniyor", train)
