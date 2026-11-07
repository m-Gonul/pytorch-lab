from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        ensure(need(ns, "tconv_out_size")(7, 4, 2, 1) == 14 and ns["tconv_out_size"](7, 3, 2, 1, 1) == 14, "çıkış boyutu (output_padding dahil)")
        close(need(ns, "conv_transpose1d")([1, 2, 3], [1, 2, 1], 2), [1, 2, 3, 4, 5, 6, 3], what="1B scatter")
        close(ns["conv_transpose1d"]([1, 2, 3], [0.5, 1, 0.5], 2, 0)[:5], [0.5, 1, 1.5, 2, 2.5], what="[½, 1, ½] = doğrusal interpolasyon")
        close(ns["conv_transpose1d"]([1, 2, 3], [1, 2, 1], 2, 1), [2, 3, 4, 5, 6], what="dolgu kırpması")
        y = need(ns, "conv_transpose2d")([[1, 2], [3, 4]], [[1, 1], [1, 1]], 2)
        close(y, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], what="k = s = 2: en yakın komşu büyütme")
        z = ns["conv_transpose2d"]([[1, 0], [0, 0]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1)
        ensure(z[1][1] == 5 and len(z) == 4, "tek piksel → çekirdeğin kopyası")
        ensure(need(ns, "unet_sizes")(572)[-1] == 388 and ns["unet_sizes"](572)[4] == 32, "U-Net 572 → 388")
        d, i = need(ns, "dice_iou")([[1, 1, 1], [1, 1, 1], [0, 0, 0]], [[0, 1, 1], [0, 1, 1], [0, 1, 0]])
        close((d, i), (8 / 11, 4 / 7), what="Dice ve IoU")
        rows = []
        for k, s in ((2, 2), (3, 2), (4, 2), (3, 1), (5, 3), (6, 3)):
            c = need(ns, "overlap_counts")(12, k, s)[k:-k]
            rows.append([k, s, "".join(str(v) for v in c[:8]), "düzgün" if min(c) == max(c) else "dama tahtası"])
        R_.table("1B transposed conv: iç bölgede çıkış başına katkı sayısı", ["k", "s", "desen", ""], rows,
                 note="k, s'nin katı olmadığında katkı sayısı periyodik değişir: aynı ağırlıklar bile 'çizgili' çıktı üretir (2B'de dama tahtası). Öğrenme bunu telafi edebilir ama genellikle tamamen etmez.")
    R_.test("transposed conv ve ölçütler", t)
