from labcheck import need, ensure


def check(ns, R_):
    def task():
        import numpy as np
        make = need(ns, "make_task")
        backup = need(ns, "expected_backup")
        evaluate = need(ns, "evaluate_policy")
        learn = need(ns, "learn")
        for b in (1, 3):
            nxt, rew = make(11, b, 800001)
            ensure(nxt.shape == rew.shape == (11, 2, b), "dal şekli")
            ensure(all(len(set(row)) == b for row in nxt.reshape(-1, b)), "yerine koymadan ardıllar")
            Q = np.arange(22, dtype=float).reshape(11, 2)/22
            expected = .9*np.mean(rew[0, 0]+Q[nxt[0, 0]].max(axis=1))
            ensure(abs(backup(Q, nxt, rew, 0, 0)-expected) < 1e-12, "tam beklenen hedef")
            actions = Q.argmax(axis=1)
            P = np.zeros((11, 11))
            r = .9*rew[np.arange(11), actions].mean(axis=1)
            for i in range(11):
                for z in nxt[i, actions[i]]:
                    P[i, z] += .9/b
            exact = np.linalg.solve(np.eye(11)-P, r)
            V, residual = evaluate(Q, nxt, rew)
            ensure(np.max(abs(V-exact)) < 1e-9 and residual < 1e-10, "politika gerçek değerleri")
            for mode in ("uniform", "trajectory"):
                result = learn(nxt, rew, 70, mode, 1, [0, 10, 70])
                ensure(result["backups"] == 70 and len(result["values"]) == 3, "gerçek güncelleme bütçesi")
                ensure(max(result["residuals"]) < 1e-10, "Q yerine politika değerlendirmesi")
        # Cyclic uniform order: the first backup changes only Q(0,0).
        nxt, rew = make(7, 1)
        result = learn(nxt, rew, 1, "uniform")
        ensure(np.count_nonzero(result["Q"]) == 1, "yerinde döngüsel tarama")
    R_.test("8.8: model, expected yedek ve bağımsız doğrusal politika çözümü", task)
