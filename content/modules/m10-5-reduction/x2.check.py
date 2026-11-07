import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="warp_serialization"))

    def t():
        close(need(ns, "warp_serialization")(np.stack([np.zeros(32), np.arange(32), np.arange(32) % 4])), np.array([32, 1, 8]), what="warp başına sıralama")
        rng = np.random.default_rng(0)
        imgs = {"tekdüze": rng.integers(0, 256, 65536), "doğal benzeri (N(128, 20))": np.clip(rng.normal(128, 20, 65536), 0, 255).astype(int),
                "düzgün bölgeli (32'lik koşular)": np.repeat(rng.integers(0, 256, 2048), 32), "tek renk": np.full(65536, 42)}
        rows = []
        for name, im in imgs.items():
            g, _ = need(ns, "cost")(im, 256, False)
            sh, merge = ns["cost"](im, 256, True, 256)
            rows.append([name, g, f"{g / (65536 // 32):.1f}", f"{sh} + {merge}"])
        ensure(rows[3][1] == 65536 and rows[0][1] < 3 * 65536 // 32, "tek renkte tam sıralama")
        R_.table("65 536 piksel, warp başına atomik sıralama adımları", ["görüntü", "global atomik adımı", "warp başına", "özel: blok içi adım + global birleştirme"], rows,
                 note="Doğal görüntülerde komşu pikseller benzer: warp içi çakışma artar. Özelleştirme sıralamayı ortadan kaldırmaz ama onu çip üstüne taşır (paylaşımlı bellek atomikleri ~10× hızlı) ve global atomikleri blok başına 256'ya indirir. Koşularda birleştirme (aggregation) tek atomikle k sayımı yazar.")
    R_.test("atomik çakışma", t)
