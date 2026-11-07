import torch
from labcheck import close, need, ensure


def check(ns, R_):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)

    def cp():
        for n in (1, 255, 256, 1000):
            x = torch.randn(n, generator=g).to(dev)
            close(need(ns, "copy")(x).cpu().numpy(), x.cpu().numpy(), what=f"kopya n = {n}")
    R_.test("copy", cp)

    def tr():
        for M, N in ((16, 16), (37, 50), (64, 96), (1, 7)):
            x = torch.randn(M, N, generator=g).to(dev)
            close(need(ns, "transpose_naive")(x).cpu().numpy(), x.t().cpu().numpy(), what=f"naif {M}×{N}")
            close(need(ns, "transpose_tiled")(x).cpu().numpy(), x.t().cpu().numpy(), what=f"karolu {M}×{N}")
            close(ns["transpose_tiled"](x, 8, 32).cpu().numpy(), x.t().cpu().numpy(), what=f"karolu 8×32 blok {M}×{N}")
        if dev == "cpu":
            R_.note("CPU yorumlayıcısında doğruluk sınandı; hız için GPU gerekir. Triton'da tek bir [BM, BN] bloğu yükleyip tl.trans ile yazmak, CUDA'da paylaşımlı bellek karosuyla yaptığımız işi derleyiciye bırakır: Triton yerleşimi (layout) ve gerekirse paylaşımlı bellek dönüşümünü kendisi seçer.")
    R_.test("transpozlar (kenar blokları)", tr)
