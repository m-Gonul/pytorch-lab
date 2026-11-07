import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)

    def ad():
        for n in (1, 255, 256, 1000, 4097):
            x, y = torch.randn(n, generator=g).to(dev), torch.randn(n, generator=g).to(dev)
            close(need(ns, "add")(x, y), x + y, what=f"n = {n}")
    R.test("add (farklı n, kenar blokları)", ad)

    def sm():
        for shape in ((64, 100), (7, 1), (3, 257)):
            x = (torch.randn(*shape, generator=g) * 5 - 20).to(dev)
            close(need(ns, "softmax")(x), torch.softmax(x, -1), atol=1e-6, what=f"{shape}")
    R.test("softmax (negatif satırlar, 2'nin kuvveti olmayan sütunlar)", sm)

    def strided():
        x = torch.randn(10, 64, generator=g).to(dev)[:, :50]
        close(need(ns, "softmax")(x), torch.softmax(x, -1), atol=1e-6, what="bitişik olmayan girdi")
    R.test("dilimlenmiş girdi", strided)
    if dev == "cuda" and "softmax" in ns:
        x = torch.randn(4096, 1024, device="cuda")
        t1 = bench(lambda: ns["softmax"](x), repeat=10)
        t2 = bench(lambda: torch.softmax(x, -1), repeat=10)
        R.table("4096 × 1024 softmax (GPU)", ["yöntem", "süre"], [["senin Triton", fmt_time(t1)], ["torch.softmax", fmt_time(t2)]])
    else:
        R.note("CPU yorumlayıcısında çalıştı: sonuçlar doğru, süreler anlamsız. Hız için Colab'da GPU çalışma zamanı kullan.")
