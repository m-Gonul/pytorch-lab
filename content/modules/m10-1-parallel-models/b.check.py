import random
from labcheck import close, no_imports, need, ensure


def rand_dag(n, seed):
    rng = random.Random(seed)
    dur = {i: rng.randint(1, 9) for i in range(n)}
    preds = {i: [j for j in range(max(0, i - 6), i) if rng.random() < 0.3] for i in range(n)}
    return dur, preds


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def laws():
        close(need(ns, "amdahl")(0.9, 8), 1 / 0.2125, what="Amdahl")
        close(need(ns, "gustafson")(0.9, 8), 7.3, what="Gustafson")
        close(need(ns, "karp_flatt")(ns["amdahl"](0.95, 16), 16), 0.05, what="Karp–Flatt Amdahl'ı geri verir")
        close(need(ns, "pipeline_makespan")([2, 3, 5, 2, 3], 100), 510, what="boru hattı")
        close(ns["pipeline_makespan"]([1, 1, 1], 1), 3, what="tek öğe")
    R_.test("yasalar ve boru hattı", laws)

    def sched():
        rows = []
        for n, seed in ((30, 0), (80, 1), (200, 2)):
            dur, preds = rand_dag(n, seed)
            W, S = need(ns, "work_span")(dur, preds)
            ensure(W == sum(dur.values()), "iş = süre toplamı")
            r = [n, W, S, f"{W / S:.1f}"]
            for p in (1, 2, 4, 8):
                T = need(ns, "list_schedule")(dur, preds, p)
                ensure(max(W / p, S) - 1e-9 <= T <= W / p + S + 1e-9, f"Brent sınırları (n = {n}, p = {p})")
                if p == 1:
                    ensure(T == W, "tek işlemcide T = W")
                r.append(f"{T:g}")
            rows.append(r)
        R_.table("Rastgele görev grafikleri, alt-seviye öncelikli liste zamanlama", ["görev", "W", "S∞", "W/S∞", "T₁", "T₂", "T₄", "T₈"], rows,
                 note="p, paralellik W/S∞'u aştığında hızlanma doyar: kritik yol sınırı. Brent'in üst sınırı her açgözlü zamanlayıcı için geçerlidir.")
    R_.test("iş–yol ve zamanlama", sched)
