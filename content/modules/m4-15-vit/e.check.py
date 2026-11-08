import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labcheck import close, need, ensure, fmt_time


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 384, 16, 16)
        x = torch.randn(8, 3, 224, 224)
        W, b = conv.weight.flatten(1), conv.bias
        with torch.no_grad():
            ref = conv(x).flatten(2).transpose(1, 2)
            close(need(ns, "patch_embed_unfold")(x, W, b, 16).numpy(), ref.numpy(), atol=1e-4, what="unfold")
            close(need(ns, "patch_embed_reshape")(x, W, b, 16).numpy(), ref.numpy(), atol=1e-4, what="reshape")
            tm = {}
            for name, f in (("conv", lambda: conv(x)), ("unfold", lambda: ns["patch_embed_unfold"](x, W, b, 16)), ("reshape", lambda: ns["patch_embed_reshape"](x, W, b, 16))):
                f()
                t0 = time.perf_counter()
                for _ in range(3):
                    f()
                tm[name] = (time.perf_counter() - t0) / 3
            rows = []
            layer = nn.TransformerEncoderLayer(384, 6, 1536, 0.0, activation="gelu", batch_first=True, norm_first=True).eval()
            for P in (32, 16, 8):
                T = (224 // P) ** 2 + 1
                z = torch.randn(4, T, 384)
                layer(z)
                t0 = time.perf_counter()
                layer(z)
                tl = time.perf_counter() - t0
                lin = 4 * T * 384 * 384 + 2 * T * 384 * 1536
                att = 2 * T * T * 384
                rows.append([P, T, fmt_time(tl), f"{att / (lin + att):.0%}"])
        R_.note("Yama embedding'i (batch 8, 224², D = 384): " + ", ".join(f"{k} {fmt_time(v)}" for k, v in tm.items()) + ". Üçü aynı hesap; reshape kopyasız görünüm + tek matmul.")
        R_.table("Bir ViT-S katmanı (D = 384, 6 baş), batch 4, 224 × 224 (CPU)", ["P", "token", "süre", "attention FLOP payı"], rows,
                 note="Yama boyutu yarıya inince token 4×, attention maliyeti 16× artar. P = 8'de attention baskın hale gelir: küçük yama = ince ayrıntı ama karesel bedel. Swin'in pencere attention'ı (x3) bu bedeli doğrusala indirir.")
    R_.test("yama embedding'i ve yama boyutu", t)
