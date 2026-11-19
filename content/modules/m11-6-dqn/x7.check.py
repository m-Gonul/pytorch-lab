from labcheck import need, ensure, close


def check(ns, R):
    def one():
        f = need(ns, "baird_q")
        r = f(1, .01, 1)
        ensure(r["visits"][4][0] == 1 and r["actual_steps"] == 1, "ilk davranış geçişi")
        expected = [[1.]*6+[10.,1.] for _ in range(2)]
        expected[0][4] -= .0006
        expected[0][7] -= .0003
        close(r["Q"], expected, what="yalnız mevcut eylemin aktif iki özelliği")
        r = f(100000, .01, 1)
        ensure(sum(sum(v) for v in r["visits"]) == 100000 and min(min(v) for v in r["visits"]) > 0, "bütün çiftler ziyaret edildi")
        ensure(len(r["Q"]) == 2 and all(len(v) == 8 for v in r["Q"]), "iki bağımsız eylem bloğu")
        close(r["norms"][-1], 3.35030946352351e16, rtol=1e-8, what="bağımsız C++ son norm")
        ensure(r["norms"][-1] > 1e12 and r["points"][-1] == 100000, "büyüme gerçek bütçede ölçüldü")
        R.table("Q-learning: 16 ağırlık ve sabit davranış", ["adım", "ağırlık normu"], [[r["points"][i], f"{r['norms'][i]:.7g}"] for i in (0, 2, 20, 200)], note="Başlangıç iki eylemde aynı; gerçek Q*0. α0.01,γ0.99 ve tohum1 kendi ölçüm koşullarımızdır.")
    R.test("Baird Q-learning: koşullu eylem güncellemesi ve büyüme", one)
