from labcheck import need, ensure


def check(ns, R_):
    def trial(tagged, seed, W=3, rounds=4, nodes=("A", "B", "C")):
        mem = {"top": (None, 0), "next": {n: None for n in nodes}, "retries": 0}
        for n in reversed(nodes):
            mem["next"][n] = mem["top"][0]
            mem["top"] = (n, 0)
        steps = ns["schedule"]([ns["worker"](mem, rounds, tagged) for _ in range(W)], seed)
        return ns["stack_ok"](mem, list(nodes)), mem["retries"], steps

    def t_cas():
        mem = {"x": (1, 0)}
        ensure(need(ns, "cas")(mem, "x", (1, 0), (2, 1)) is True and mem["x"] == (2, 1), "CAS: beklenen eşitse yaz, True")
        ensure(ns["cas"](mem, "x", (1, 0), (3, 3)) is False and mem["x"] == (2, 1), "CAS: eşit değilse dokunma, False")
        mem = {"top": (None, 0), "next": {"A": None, "B": None}, "retries": 0}
        for g in (need(ns, "push")(mem, "A", True), ns["push"](mem, "B", True)):
            for _ in g:
                pass
        g = need(ns, "pop")(mem, True)
        try:
            while True:
                next(g)
        except StopIteration as e:
            ensure(e.value == "B" and mem["top"][0] == "A", "tek thread: LIFO")
    R_.test("CAS, push, pop (tek thread)", t_cas)

    def t_aba():
        rows, res = [], {}
        for tagged in (False, True):
            for W in (2, 3, 4):
                r = [trial(tagged, s, W=W) for s in range(600)]
                bad = sum(not ok for ok, _, _ in r) / len(r)
                res[(tagged, W)] = bad
                rows.append(["etiketli (ptr, sayaç)" if tagged else "yalın işaretçi", W, f"{bad:.3f}", f"{sum(x for _, x, _ in r) / len(r) / (W * 4 * 2):.2f}"])
        ensure(res[(False, 3)] > 0 and all(res[(True, w)] == 0 for w in (2, 3, 4)), f"ABA yalın CAS'ta görülmeli, etiketlide asla {res}")
        R_.table("Treiber yığını: 3 düğüm, her işçi 4 kez pop + aynı düğümü push (600 rastgele zamanlama)", ["CAS karşılaştırması", "işçi", "bozulan yığın oranı", "işlem başına yeniden deneme"], rows,
                 note="ABA: thread 1 top = A ve A.next = B'yi okur, kesilir; thread 2 A'yı ve B'yi çıkarır, A'yı geri koyar. top yine A olduğu için thread 1'in CAS'ı başarılı olur ve top'u artık yığında olmayan B'ye çevirir. Sürüm sayacı (etiket) her başarılı CAS'ta arttığı için 'aynı görünen' durum ayırt edilir (çift genişlikli CAS ya da LL/SC ile). Kilitsiz algoritma sistem genelinde ilerleme garanti eder ama tek bir thread'in sürekli yeniden denemesini (açlık) engellemez.")
    R_.test("ABA sorunu", t_aba)
