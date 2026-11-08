from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        ensure(need(ns, "n_patches")(224, 224, 16) == 196 and ns["n_patches"](384, 384, 16) == 576, "yama sayısı")
        ensure(need(ns, "vit_params")(768, 12, 3072, 16, 3, 224, 1000) == 86567656, "ViT-B/16")
        img = [[r * 4 + c for c in range(4)] for r in range(4)]
        ensure(need(ns, "patchify")(img, 2) == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], "yamalar row-major")
        ensure(need(ns, "attn_flops")(197, 768) == 2 * 197 * 197 * 768, "attention FLOP")
        rows = []
        for name, D, L, M in (("ViT-B", 768, 12, 3072), ("ViT-L", 1024, 24, 4096), ("ViT-H", 1280, 32, 5120)):
            for P in (32, 16, 14):
                T = ns["n_patches"](224, 224, P) + 1
                lin = L * (4 * T * D * D + 2 * T * D * M)
                att = L * ns["attn_flops"](T, D)
                rows.append([f"{name}/{P}", T, f"{ns['vit_params'](D, L, M, P, 3, 224, 1000) / 1e6:.0f} M", f"{(lin + att) / 1e9:.1f}", f"{att / (lin + att):.0%}"])
        R_.table("224 × 224'te ViT ailesi", ["model", "token", "parametre", "GMAC", "attention payı"], rows,
                 note="Parametre sayısı yama boyutuna neredeyse bağlı değil (yalnız yama embedding'i ve konumlar), ama hesap token sayısıyla büyür: P = 16 → 14'te token %31 artar. 224'te hesabın çoğu doğrusal katmanlarda (QKV, MLP); attention payı ancak dizi uzadıkça (yüksek çözünürlük, küçük yama) baskınlaşır.")
    R_.test("sayımlar", t)
