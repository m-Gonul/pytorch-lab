import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t_mlm():
        rng = random.Random(0)
        toks = [1] + [5 + (i % 90) for i in range(20000)] + [2]
        inp, lab = need(ns, "mlm_mask")(toks, rng, 100, 3, 4, 0.15)
        ensure(lab[0] == -100 and lab[-1] == -100 and inp[0] == 1, "özel token'lar dokunulmaz")
        sel = [i for i, l in enumerate(lab) if l != -100]
        ensure(all(lab[i] == toks[i] for i in sel), "etiket = orijinal")
        f = len(sel) / 20000
        m = sum(inp[i] == 3 for i in sel) / len(sel)
        same = sum(inp[i] == toks[i] for i in sel) / len(sel)
        ensure(abs(f - 0.15) < 0.01 and abs(m - 0.8) < 0.02 and abs(same - 0.1) < 0.02 + 0.01, f"oranlar seçim {f:.3f}, mask {m:.3f}, aynı {same:.3f}")
        ensure(all(inp[i] == toks[i] for i in range(len(toks)) if lab[i] == -100), "seçilmeyenler değişmez")
        P = need(ns, "nsp_pairs")([[i] for i in range(400)], random.Random(1))
        pos = [p for p in P if p[2]]
        ensure(all(b[0] == a[0] + 1 for a, b, _ in pos) and all(b[0] != a[0] + 1 for a, b, c in P if not c) and abs(len(pos) / len(P) - 0.5) < 0.08, "NSP çiftleri")
    R_.test("MLM ve NSP", t_mlm)

    def t_num():
        ensure(need(ns, "bert_params")(30522, 768, 12, 3072) == 109482240, "BERT-base")
        close(need(ns, "bt_loss")(2, 0.5), -math.log(1 / (1 + math.exp(-1.5))), what="Bradley–Terry")
        close(need(ns, "kl_reward")(1, -1.0, -3.0, 0.1), 0.8, what="KL cezalı ödül")
        close(need(ns, "dpo_loss")(-1.0, -5.0, -3.0, -4.0, 0.1), -math.log(1 / (1 + math.exp(-0.3))), what="DPO")
        rows = []
        for name, (d, L, ff) in (("BERT-base", (768, 12, 3072)), ("BERT-large", (1024, 24, 4096))):
            p = ns["bert_params"](30522, d, L, ff)
            rows.append([name, f"{p / 1e6:.1f} M", f"{(30522 + 514) * d / p:.0%}"])
        R_.table("BERT parametreleri", ["model", "toplam", "embedding payı"], rows,
                 note="BERT-large makalede 340M; burada 335M (MLM ve NSP başları hariç). Base'de parametrelerin ~%22'si embedding'lerde.")
    R_.test("sayımlar ve kayıplar", t_num)
