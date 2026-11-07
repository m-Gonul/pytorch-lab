import json
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = random.Random(0)
    layers = [{"op": "linear", "W": [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(4)], "b": [0.1, -0.2, 0.0, 0.3]},
              {"op": "relu"},
              {"op": "linear", "W": [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(2)], "b": [0.0, 1.0]}]

    def js():
        s = need(ns, "to_json")(layers)
        ensure(isinstance(s, str) and json.loads(s) == layers, "geçerli JSON")
        x = [0.5, -1.0, 2.0]
        close(need(ns, "run")(need(ns, "from_json")(s), x), ns["run"](layers, x))
        h = [max(0.0, sum(w * v for w, v in zip(r, x)) + b) for r, b in zip(layers[0]["W"], layers[0]["b"])]
        close(ns["run"](layers, x), [sum(w * v for w, v in zip(r, h)) + b for r, b in zip(layers[2]["W"], layers[2]["b"])])
    R.test("JSON gidiş-dönüş ve yorumlayıcı", js)

    def tr():
        f = lambda x: x * 2 if x.sum() > 0 else -x
        g = need(ns, "trace")(f, [1.0, 2.0])
        ensure([tuple(o) for o in g] == [("mul", 2)], f"graf {g}")
        out = need(ns, "replay")(g, [-1.0, -1.0])
        close(out, [-2.0, -2.0], what="donmuş dal")
        g2 = ns["trace"](f, [-1.0, -1.0])
        close(ns["replay"](g2, [-1.0, -1.0]), [1.0, 1.0], what="negatif örnekle izlenince öbür dal")
        return "izleme girdinin aldığı dalı donduruyor: x = (−1, −1) için 2x hesaplandı"
    R.test("tracing kontrol akışını donduruyor", tr)

    def chain():
        g = need(ns, "trace")(lambda x: -(3 * (x * 0.5)), [1.0])
        close(need(ns, "replay")(g, [4.0, -2.0]), [-6.0, 3.0])
    R.test("zincirlenmiş işlemler (sağdan çarpma dahil)", chain)
