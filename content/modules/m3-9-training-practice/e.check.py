from labcheck import need, ensure, bench, fmt_time


def check(ns, R_):
    def t():
        widths = [2, 8, 32, 128]
        b = need(ns, "batched_cv")(widths, 5, 150, 0)
        l = need(ns, "loop_cv")(widths, 5, 150, 0)
        ensure(set(b) == set(widths) and set(l) == set(widths), "her genişlik için sonuç")
        ensure(b[128][0] > b[2][0] + 0.2 and l[128][0] > l[2][0] + 0.2, "genişlik 2 yetersiz uyum")
        for w in widths[1:]:
            ensure(abs(b[w][0] - l[w][0]) < 0.08, f"toplu ve döngülü CV yakın (genişlik {w})")
        tb = bench(lambda: ns["batched_cv"](widths, 5, 150, 0), repeat=1, number=1)
        tl = bench(lambda: ns["loop_cv"](widths, 5, 150, 0), repeat=1, number=1)
        R_.table("digits, 5 katlı CV, tek gizli katman, 150 tam batch adım", ["genişlik", "toplu: ort. ± std", "döngü: ort. ± std"], [[w, f"{b[w][0]:.3f} ± {b[w][1]:.3f}", f"{l[w][0]:.3f} ± {l[w][1]:.3f}"] for w in widths],
                 note=f"Süre: toplu {fmt_time(tb)}, döngü {fmt_time(tl)} (20 model). Toplu sürümde genişlikler maskeyle tek tensörde; başlatmalar farklı olduğundan sonuçlar birebir değil, istatistiksel olarak uyumlu — genişlik 2 gibi kararsız modellerde başlatma farkı büyük olabilir.")
    R_.test("toplu CV", t)
