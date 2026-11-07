from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    wf = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

    def pc():
        c = need(ns, "pair_counts")({tuple(w): f for w, f in wf.items()})
        ensure(c[("e", "s")] == 9 and c[("w", "e")] == 8 and c[("l", "o")] == 7, "çift sayıları (e,s)=9, (w,e)=8, (l,o)=7")
    R.test("pair_counts", pc)

    def mv():
        v = need(ns, "merge_vocab")({("a", "a", "a"): 1, ("a", "b", "a", "a"): 2}, ("a", "a"))
        ensure(v == {("aa", "a"): 1, ("a", "b", "aa"): 2}, f"örtüşmeyen soldan-sağa birleştirme: {v}")
    R.test("merge_vocab (aaa → aa, a)", mv)

    def tr():
        m = need(ns, "train_bpe")(wf, 4)
        ensure([tuple(p) for p in m] == [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w")], f"birleştirmeler {m}")
        ensure(len(need(ns, "train_bpe")({"ab": 1}, 5)) == 1, "çift kalmayınca durmalı")
    R.test("train_bpe (eşitlik kuralı dahil)", tr)

    def enc():
        m = need(ns, "train_bpe")(wf, 4)
        ensure(list(need(ns, "encode")("lowest", m)) == ["low", "est"], "lowest → low, est")
        ensure(list(need(ns, "encode")("newer", m)) == ["n", "e", "w", "e", "r"], "newer birleşmez")
    R.test("encode", enc)
