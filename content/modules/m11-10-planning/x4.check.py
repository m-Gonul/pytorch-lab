from labcheck import need, ensure


def check(ns, R_):
    def task():
        import random
        transition = need(ns, "transition")
        choose = need(ns, "select_action")
        run = need(ns, "run_bonus")
        ensure(transition(4*9+8, 0, "shortcut", False)[0] == 44, "kapalı sağ geçit")
        ensure(transition(44, 0, "shortcut", True)[0] == 35, "açılan geçit")
        ensure(transition(36, 0, "blocking", False)[0] == 36, "önce sol kapalı")
        ensure(transition(36, 0, "blocking", True)[0] == 27, "sonra sol açık")
        ensure(choose([1., 0., 0., 0.], [0, 40000, 0, 0], .01, random.Random(0)) == 1,
               "bonus yalnız eylem seçimini değiştirebilmeli")
        # A one-step run cannot reach the goal. A selection bonus must not leak into Q.
        for variant in ("selection", "plain"):
            r = run(n_steps=1, kappa=1., n_plan=20, variant=variant)
            ensure(all(v == 0 for row in r["Q"] for v in row), "bonussuz öğrenme hedefi")
            ensure(r["cumulative"] == [0.] and set(r["last"].values()) == {1}, "gerçek saat/ödül")
        r = run(n_steps=300, change=150, n_plan=5)
        ensure(len(r["cumulative"]) == 300 and all(a <= b for a, b in zip(r["cumulative"], r["cumulative"][1:])), "birikimli gerçek ödül")
        ensure(all(1 <= t <= 300 for t in r["last"].values()), "yaş yalnız gerçek zamanda")
    R_.test("8.4: seçim bonusu, duvar değişimi ve gerçek ödül", task)
