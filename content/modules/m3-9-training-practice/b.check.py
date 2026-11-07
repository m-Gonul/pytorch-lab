from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        X, y, g = need(ns, "make_groups")(30, 4, 1)
        ensure(len(X) == 120 and len(set(g)) == 30, "boyutlar")
        tr, te = need(ns, "split_groups")(g, 0.7, 0)
        ensure(not ({g[i] for i in tr} & {g[i] for i in te}), "grup bölmesinde ortak grup olmamalı")
        tr, te = need(ns, "split_random")(120, 0.7, 0)
        ensure(sorted(tr + te) == list(range(120)), "rastgele bölme tam")
        ensure(need(ns, "nn1_predict")([[0, 0], [5, 5]], [0, 1], [[1, 1], [4, 6]]) == [0, 1], "1-NN")
        rows = []
        accs = []
        for s in range(5):
            X, y, g = ns["make_groups"](60, 5, s)
            a, b = need(ns, "accuracy_both")(X, y, g, s)
            accs.append((a, b))
            rows.append([s, f"{a:.2f}", f"{b:.2f}"])
        ra = sum(a for a, _ in accs) / 5
        gb = sum(b for _, b in accs) / 5
        ensure(ra > 0.9 and gb < 0.8, "rastgele bölme şişirilmiş, grup bölmesi gerçekçi")
        R_.table("1-NN, 60 denek × 5 tekrar ölçüm", ["tohum", "rastgele bölme", "grup (denek) bölmesi"], rows,
                 note="Rastgele bölmede test örneğinin 'ikizi' eğitimde: 1-NN onu ezberden bilir. Gerçek soru 'yeni bir denekte ne olur?' ise bölme denek düzeyinde yapılmalı (GroupKFold).")
    R_.test("grup sızıntısı", t)
