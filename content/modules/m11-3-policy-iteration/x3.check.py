import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        V, pi = need(ns, "gambler_vi")(0.4)
        ensure(V[0] == 0 and V[100] == 1 and np.all(np.diff(V) >= -1e-12), "V monoton artan")
        close(V[50], 0.4, atol=1e-9, what="V(50) = p_h (hepsini yatır)")
        close(V[25], 0.4 * 0.4, atol=1e-9, what="V(25) = p_h²")
        ensure(pi[50] == 50 and pi[25] == 25 and pi[75] == 25, f"π(50) = {pi[50]}, π(25) = {pi[25]}, π(75) = {pi[75]}")
        V2, pi2 = ns["gambler_vi"](0.55)
        ensure(np.all(pi2[1:100] == 1), "p_h > 0.5'te en iyisi hep 1 yatırmak")
        R_.table("Kumarbaz problemi, p_h = 0.4", ["sermaye", "V", "π (bahis)"], [[s, f"{V[s]:.4f}", pi[s]] for s in (1, 10, 12, 25, 37, 50, 51, 75, 99)],
                 note="p_h < 0.5'te oyun aleyhte: az sayıda büyük bahisle 'cesur oyun' en iyisidir; politika 12.5, 25, 50 gibi noktalarda dikenlidir.")
    R_.test("kumarbaz problemi", t)
