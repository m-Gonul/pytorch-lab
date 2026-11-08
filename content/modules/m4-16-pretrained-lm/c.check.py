import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: ([max_loops(src, 0, func=f) for f in ("mlm_mask_batch", "elmo_mix", "seq_logprob", "dpo_batch")], max_loops(src, 1, func="fit_reward")))
    rng = np.random.default_rng(0)

    def t():
        X = rng.integers(4, 100, (200, 128))
        X[:, 0] = 1
        inp, lab = need(ns, "mlm_mask_batch")(X, np.random.default_rng(1), 100, 3, 4)
        sel = lab != -100
        ensure(not sel[:, 0].any() and abs(sel.mean() - 0.15 * 127 / 128) < 0.01, "seçim oranı")
        ensure(abs((inp[sel] == 3).mean() - 0.8) < 0.02 and np.all(inp[~sel] == X[~sel]) and np.all(lab[sel] == X[sel]), "80/10/10")
        H = rng.normal(size=(3, 5, 4))
        close(need(ns, "elmo_mix")(H, np.array([0.0, np.log(3), 0.0]), 2.0), 2 * (0.2 * H[0] + 0.6 * H[1] + 0.2 * H[2]), what="ELMo karışımı")
        logits = rng.normal(size=(2, 4, 6))
        y = np.array([[1, 2, 3, 4], [5, 0, 1, 2]])
        m = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        close(need(ns, "seq_logprob")(logits, y, m), [lp[0, 2, 3] + lp[0, 3, 4], lp[1, 1, 0] + lp[1, 2, 1]], what="yanıt log olasılığı")
        L, acc, mg = need(ns, "dpo_batch")(np.array([-1.0, -2.0]), np.array([-5.0, -1.0]), np.array([-3.0, -2.0]), np.array([-4.0, -2.0]), 0.1)
        close(L, np.mean([np.log1p(np.exp(-0.3)), np.log1p(np.exp(0.1))]), what="DPO kaybı")
        ensure(acc == 0.5, "örtük ödül doğruluğu")
        wtrue = rng.normal(size=8)
        Fa, Fb = rng.normal(size=(3000, 8)), rng.normal(size=(3000, 8))
        pa = 1 / (1 + np.exp(-(Fa - Fb) @ wtrue))
        win = rng.random(3000) < pa
        Fw, Fl = np.where(win[:, None], Fa, Fb), np.where(win[:, None], Fb, Fa)
        w = need(ns, "fit_reward")(Fw, Fl, 0.5, 500)
        cos = w @ wtrue / np.linalg.norm(w) / np.linalg.norm(wtrue)
        ensure(cos > 0.95, f"ödül yönü kurtarılır (cos {cos:.3f})")
        rows = []
        for n in (100, 300, 1000, 3000):
            wn = ns["fit_reward"](Fw[:n], Fl[:n], 0.5, 500)
            rows.append([n, f"{wn @ wtrue / np.linalg.norm(wn) / np.linalg.norm(wtrue):.3f}", f"{np.linalg.norm(wn) / np.linalg.norm(wtrue):.2f}"])
        R_.table("Bradley–Terry ödül modeli: yalnız ikili tercihlerden doğrusal ödülü kurtarmak", ["tercih çifti", "cos(ŵ, w)", "‖ŵ‖/‖w‖"], rows,
                 note="Ödül modeli yalnız farkları görür: r'ye eklenen bir sabit tercihleri değiştirmez (ödül ölçeğe kadar değil, öteleme dışında belirlidir). Az veride ŵ'nin normu şişer (ayrılabilir tercihler); pratikte ödül modelleri erken durdurma ve normalizasyonla eğitilir.")
    R_.test("maskeleme, ELMo, DPO, ödül modeli", t)
