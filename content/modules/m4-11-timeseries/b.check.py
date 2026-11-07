from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t_diff():
        ensure(need(ns, "difference")([3, 5, 7, 9, 11]) == [2, 2, 2, 2], "doğrusal trend")
        ensure(ns["difference"]([1, 4, 9, 16, 25, 36], 2) == [2, 2, 2, 2], "ikinci dereceden trend")
        close(need(ns, "undifference")([3, 5, 7, 9, 11], [2, 2], 1), [13, 15], what="ARIMA d = 1")
        close(ns["undifference"]([1, 4, 9, 16, 25, 36], [2, 2, 2], 2), [49, 64, 81], what="d = 2: kareler sürer")
        p = need(ns, "arma_predict")([1.0, 2.0, 0.5], [0.5], [0.4])
        close(p, [0, 0.9, 1.44, 0.25 + 0.4 * (0.5 - 1.44)], what="ARMA(1,1) özyinelemesi")
    R_.test("fark alma ve ARMA", t_diff)

    def t_yw():
        r = need(ns, "ar_acf")([0.6, 0.3], 5)
        close(r[:3], [1, 0.6 / 0.7, 0.6 * 0.6 / 0.7 + 0.3], what="AR(2) ρ")
        close(ns["ar_acf"]([0.8], 4), [1, 0.8, 0.64, 0.512, 0.4096], what="AR(1): ρ_k = φ^k")
        r3 = ns["ar_acf"]([0.3, -0.2, 0.4], 6)
        close(r3[1], 0.3 * r3[0] - 0.2 * r3[1] + 0.4 * r3[2], what="AR(3) Yule–Walker k = 1")
        close(r3[5], 0.3 * r3[4] - 0.2 * r3[3] + 0.4 * r3[2], what="AR(3) özyineleme")
    R_.test("Yule–Walker", t_yw)

    def t_cnt():
        P = need(ns, "rnn_params")
        ensure(P([1]) == 3 and P([16, 16, 1]) == 834 and P([16, 16]) + 136 == 952, "Keras sayımı")
        ensure(P([16, 16, 1], double_bias=True) == 834 + 33, "PyTorch: katman başına +h")
        ensure(need(ns, "receptive_field")(2, [2 ** i for i in range(10)] * 2) == 2047, "WaveNet")
        rows = [[k, str(d), ns["receptive_field"](k, d)] for k, d in ((2, [1, 1, 1, 1]), (2, [1, 2, 4, 8]), (3, [1, 2, 4, 8]), (2, [2 ** i for i in range(10)]), (2, [2 ** i for i in range(10)] * 3))]
        R_.table("Nedensel konvolüsyon yığınlarının receptive field'ı", ["k", "genişletmeler", "RF"], [[a, b if len(b) < 30 else "1…512 ×" + str(len(eval(b)) // 10), c] for a, b, c in rows],
                 note="Genişletmesiz yığında RF katman sayısıyla doğrusal, ikiye katlanan genişletmelerle üstel büyür: 10 katmanlık 1…512 bloğu 1024 adım görür. WaveNet bu bloğu birkaç kez yığar.")
    R_.test("parametre ve RF sayımı", t_cnt)
