import math
from labcheck import close, no_imports, need, ensure


TREE = {(0,): {"A": 0.6, "B": 0.4}, (0, "A"): {"x": 0.4, "y": 0.3, "z": 0.3}, (0, "B"): {"x": 0.9, "y": 0.1}}


def tree_step(prefix):
    if prefix in TREE:
        return TREE[prefix]
    return {"<eos>": 1.0}


def _spread(d, prefix, n):
    rest = (1 - sum(d.values())) / n
    d.update({f"{prefix}{i}": rest for i in range(n)})
    return d


HOW = {(): _spread({"How": 0.75, "What": 0.03, "You": 0.01}, "x", 21), ("How",): _spread({"will": 0.36, "are": 0.32, "do": 0.16}, "h", 8),
       ("What",): {"are": 0.5, "<eos>": 0.5}, ("How", "will"): _spread({"you": 0.074}, "w", 20), ("How", "are"): _spread({"you": 5 / 12}, "a", 30),
       ("How", "do"): _spread({"you": 2 / 3}, "d", 20), ("How", "are", "you"): _spread({"<eos>": 0.6, "doing": 0.3}, "r", 10),
       ("How", "do", "you"): {"do": 0.875, "<eos>": 0.125}}


def how_step(prefix):
    return HOW.get(tuple(prefix[1:]), {"<eos>": 1.0})


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t_tree():
        s, lp = need(ns, "greedy_decode")(tree_step, 0, "<eos>", 5)
        ensure(s == ["A", "x", "<eos>"], f"açgözlü {s}")
        close(math.exp(lp), 0.24, what="açgözlü olasılık")
        res = need(ns, "beam_search")(tree_step, 0, "<eos>", 2, 5)
        ensure(res[0][0] == ["B", "x", "<eos>"], f"beam {res[0][0]}")
        close(math.exp(res[0][1]), 0.36, what="beam olasılığı")
    R_.test("açgözlü vs beam", t_tree)

    def t_how():
        g = ns["greedy_decode"](how_step, "<s>", "<eos>", 6)[0]
        rows = [["açgözlü", " ".join(g), f"{math.exp(ns['greedy_decode'](how_step, '<s>', '<eos>', 6)[1]):.3f}"]]
        for k, a in ((1, 0.0), (3, 0.0), (3, 1.0)):
            r = ns["beam_search"](how_step, "<s>", "<eos>", k, 6, a)
            rows.append([f"beam k = {k}, α = {a}", " ".join(r[0][0]), f"{math.exp(r[0][1]):.3f}"])
        ensure(g == ["How", "will", "you", "<eos>"], f"açgözlü 'How will you' tuzağına düşer: {g}")
        r3 = ns["beam_search"](how_step, "<s>", "<eos>", 3, 6, 1.0)
        ensure(r3[0][0][:3] == ["How", "are", "you"] or r3[0][0][:3] == ["How", "do", "you"], f"beam + normalizasyon: {r3[0][0]}")
        ensure(ns["beam_search"](how_step, "<s>", "<eos>", 1, 6)[0][0] == g, "k = 1 = açgözlü")
        R_.table("'Comment vas-tu?' oyuncak modeli", ["çözümleyici", "çıktı", "olasılık"], rows,
                 note="Normalizasyonsuz beam search ('How <eos>' gibi) kısa hipotezleri kayırır: her token olasılığı 1'den küçük bir çarpandır. α ile log olasılığı uzunluğa bölmek (Wu ve ark. 2016: ((5 + L)/6)^α) bu yanlılığı dengeler.")
    R_.test("Géron örneği", t_how)

    def t_pad():
        p, m = need(ns, "pad_batch")([[5, 6, 7], [8]], 0)
        ensure(p == [[5, 6, 7], [8, 0, 0]] and m == [[True] * 3, [True, False, False]], "dolgu ve maske")
        ensure(need(ns, "shift_right")([4, 5, 2], 1) == [1, 4, 5], "öğretmen zorlaması girdisi")
    R_.test("dolgu ve öğretmen zorlaması", t_pad)
