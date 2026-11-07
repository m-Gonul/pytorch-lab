from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "inverted_residual_cost")(32, 28, 6) == (32 * 192 * 2 + 9 * 192, 28 * 28 * (32 * 192 * 2 + 9 * 192)), "ters artık")
        ensure(need(ns, "bottleneck_cost")(256, 14, 4) == (256 * 64 * 2 + 9 * 64 * 64, 14 * 14 * (256 * 64 * 2 + 9 * 64 * 64)), "darboğaz")
        rows = []
        for t_ in (1, 2, 4, 6, 8):
            p, m = ns["inverted_residual_cost"](64, 28, t_)
            rows.append([t_, p, f"{m / 1e6:.1f} M"])
        rows.append(["ResNet darboğazı (r = 4)", ns["bottleneck_cost"](64, 28, 4)[0], f"{ns['bottleneck_cost'](64, 28, 4)[1] / 1e6:.1f} M"])
        R_.table("c = 64, 28×28: ters artık bloğun genişleme oranı t", ["t", "parametre", "MAC"], rows,
                 note="ResNet darboğazı kanalları ortada daraltır (geniş → dar → geniş); MobileNetV2 tersini yapar (dar → geniş → dar) ve pahalı uzamsal işi ucuz derinlemesine konvolüsyona bırakır. Maliyet t ile doğrusal; son 1×1'den sonra ReLU yok (doğrusal darboğaz: dar uzayda bilgi kaybını önler).")
    R_.test("ters artık", t)
