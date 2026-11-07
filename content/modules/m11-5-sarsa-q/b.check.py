from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def env():
        f = need(ns, "cliff_step")
        ensure(f(36, 1) == (36, -100.0, False), "başlangıçtan sağa: uçurum")
        ensure(f(36, 0) == (24, -1.0, False) and f(36, 2) == (36, -1.0, False) and f(36, 3) == (36, -1.0, False), "yukarı / sınırlar")
        ensure(f(35, 2) == (47, -1.0, True), "hedef")
        ensure(f(0, 0) == (0, -1.0, False), "köşe")
    R_.test("cliff_step", env)

    def online():
        S = Qm = 0
        for seed in range(10):
            S += sum(need(ns, "sarsa")(500, 0.5, 0.1, seed)[1][-100:]) / 100
            Qm += sum(need(ns, "qlearning")(500, 0.5, 0.1, seed)[1][-100:]) / 100
        S /= 10; Qm /= 10
        ensure(S > Qm + 5, f"eğitim sırasında SARSA daha çok ödül toplamalı ({S:.1f} vs {Qm:.1f})")
        gq = [need(ns, "greedy_return")(ns["qlearning"](500, 0.5, 0.1, s)[0]) for s in range(10)]
        gs = [ns["greedy_return"](ns["sarsa"](1000, 0.1, 0.1, s)[0]) for s in range(10)]
        ensure(sum(g == -13 for g in gq) >= 9, "Q-öğrenme açgözlü yolu: kenardan, −13")
        ensure(sum(-21 <= g <= -15 for g in gs) >= 8, "SARSA açgözlü yolu: güvenli, −15…−21")
        R_.table("Uçurum yürüyüşü, 10 deney", ["yöntem", "son 100 bölümde ort. getiri (α = 0.5)", "açgözlü yolun getirisi"],
                 [["SARSA", f"{S:.1f}", ", ".join(str(int(g)) for g in gs)], ["Q-öğrenme", f"{Qm:.1f}", ", ".join(str(int(g)) for g in gq)]],
                 note="Q-öğrenme optimal (kenar) yolu öğrenir ama ε-açgözlü davranırken uçuruma düşer; SARSA keşfi hesaba katarak uzak yolu seçer (Sutton Şekil 6.4). SARSA'nın açgözlü yolu α = 0.1, 1000 bölümle.")
    R_.test("SARSA ve Q-öğrenme", online)
