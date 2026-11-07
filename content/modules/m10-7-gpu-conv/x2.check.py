from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        s = need(ns, "config_stats")(32, 32, 32, 1)
        close(s["ai_global"], 8.0, what="karo yoğunluğu")
        ensure(s["threads"] == 1024 and s["ai_smem"] == 0.25, "r = 1: her FMA iki paylaşımlı okuma")
        cands = [(BM, BN, BK, r) for BM in (32, 64, 128, 256) for BN in (32, 64, 128, 256) for BK in (8, 16, 32) for r in (1, 2, 4, 8)]
        best = need(ns, "best_config")(cands)
        ensure(best is not None and need(ns, "config_stats")(*best)["smem_bytes"] <= 48 * 1024, "sınırlar içinde")
        rows = []
        for c in [(32, 32, 32, 1), (64, 64, 16, 4), (128, 128, 8, 8), best]:
            st = ns["config_stats"](*c)
            rows.append([str(c), st["threads"], st["smem_bytes"], st["regs_acc"], f"{st['ai_global']:.1f}", f"{st['ai_smem']:.2f}"])
        R_.table("Matmul blok yapılandırmaları (fp32)", ["(BM, BN, BK, r)", "thread", "smem B", "biriktirici/thread", "FLOP/B (global)", "FLOP/B (paylaşımlı)"], rows,
                 note="Paylaşımlı bellek de bir darboğazdır: r = 1'de her FMA için iki paylaşımlı okuma. r × r yazmaç karosu bunu r kat azaltır — cuBLAS benzeri çekirdeklerin (Boehm'in yazısı) asıl sırrı.")
    R_.test("yapılandırma uzayı", t)
