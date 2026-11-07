import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def ref_conv(img, k):
    K = k.shape[0]
    acc = F.conv2d(img.float()[None, None], k.float()[None, None], padding=K // 2)[0, 0]
    return torch.clamp((acc + 0.5).floor(), 0, 255).to(torch.uint8)


def check(ns, R_):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)

    def cv():
        img = torch.randint(0, 256, (37, 45), dtype=torch.uint8, generator=g)
        for K in (3, 7, 11):
            gk = torch.exp(-((torch.arange(K) - K // 2).float() ** 2) / (K / 2) ** 2)
            k = torch.outer(gk, gk)
            k = k / k.sum()
            out = need(ns, "conv2d")(img.to(dev), k).cpu()
            d = (out.int() - ref_conv(img, k).int()).abs()
            ensure(out.dtype == torch.uint8 and int(d.max()) <= 1 and float((d == 0).double().mean()) > 0.99, f"K = {K}")
        sharp = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
        d = (ns["conv2d"](img.to(dev), sharp, 8, 16).cpu().int() - ref_conv(img, sharp).int()).abs()
        ensure(int(d.max()) <= 1, "keskinleştirme: [0, 255]'e kırpma, farklı karo boyutu")
    R_.test("Triton konvolüsyon (uint8 → fp32 → uint8)", cv)

    def mm():
        for (M, K, N) in ((64, 64, 64), (50, 70, 33), (1, 5, 3), (130, 40, 70)):
            a = torch.randn(M, K, generator=g).to(dev)
            b = torch.randn(K, N, generator=g).to(dev)
            close(need(ns, "matmul")(a, b).cpu().numpy(), (a @ b).cpu().numpy(), atol=1e-3, what=f"{M}×{K} @ {K}×{N}")
        ai = torch.randint(-50, 50, (40, 30), generator=g, dtype=torch.int32)
        bi = torch.randint(-50, 50, (30, 20), generator=g, dtype=torch.int32)
        close(ns["matmul"](ai.to(dev), bi.to(dev)).cpu().numpy(), (ai.float() @ bi.float()).numpy(), atol=1e-3, what="int32 girdi, fp32 hesap")
        R_.note("Gruplanmış sıra: ardışık programlar GROUP_M satır bloğunu paylaşır; GPU'da aynı anda çalışan programlar A'nın aynı satırlarını ve B'nin aynı sütunlarını L2'den yeniden kullanır (Triton öğreticisinde %10'a varan kazanç).")
    R_.test("Triton matmul", mm)
