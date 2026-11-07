import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("warp_sectors", "warp_bank_degree", "transpose_traffic")])

    def basic():
        A = np.stack([4 * k * np.arange(32) for k in (1, 2, 4, 8, 32)])
        close(need(ns, "warp_sectors")(A), np.array([4, 8, 16, 32, 32]), what="sektörler")
        W = np.stack([k * np.arange(32) for k in (1, 2, 3, 4, 16, 32, 33)] + [np.full(32, 7)])
        close(need(ns, "warp_bank_degree")(W), np.array([1, 2, 1, 4, 16, 32, 1, 1]), what="çakışmalar")
    R_.test("warp sayımları", basic)

    def tr():
        N = 1024
        ideal = N * N * 4 // 32
        rows = []
        res = {}
        for name, kind, pad in (("kopya", "copy", 0), ("naif transpoz", "naive", 0), ("karolu, [32][32]", "tiled", 0), ("karolu, [32][33]", "tiled", 1)):
            r, w, b = need(ns, "transpose_traffic")(N, kind, 32, pad)
            res[name] = (r, w, b)
            rows.append([name, r, w, f"{(r + w) / (2 * ideal):.2f}", f"{b:.0f}" if kind == "tiled" else "—"])
        ensure(res["kopya"][:2] == (ideal, ideal), "kopya ideal")
        ensure(res["naif transpoz"][1] == 8 * ideal, "naif yazma: her thread ayrı sektör")
        ensure(res["karolu, [32][32]"][:2] == (ideal, ideal) and res["karolu, [32][32]"][2] == 32 and res["karolu, [32][33]"][2] == 1, "karolu: birleştirilmiş, dolgu çakışmayı kaldırır")
        R_.table(f"{N}×{N} float32 transpoz, global trafik (32 B sektör) ve paylaşımlı bellek çakışması", ["çekirdek", "okuma sektörü", "yazma sektörü", "trafik / ideal", "smem çakışma derecesi"], rows,
                 note="Naif transpoz 4.5× trafik üretir (yazmalarda 8×). Karolu sürüm trafiği ideale indirir ama dolgusuz karoda sütun okuması 32-yollu çakışmayla 32 kat yavaşlar; tek bir dolgu sütunu bunu çözer.")
    R_.test("transpoz trafiği", tr)
