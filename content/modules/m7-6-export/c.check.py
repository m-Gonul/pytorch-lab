import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    inits = {"W1": rng.normal(size=(8, 4)), "b1": rng.normal(size=8), "two": np.array(2.0), "W2": rng.normal(size=(8, 3)), "b2": rng.normal(size=3)}
    nodes = [
        {"op": "Gemm", "inputs": ["x", "W1", "b1"], "output": "h"},
        {"op": "Relu", "inputs": ["h"], "output": "r"},
        {"op": "Mul", "inputs": ["W2", "two"], "output": "W2s"},
        {"op": "Transpose", "inputs": ["W2s"], "output": "W2t"},
        {"op": "Gemm", "inputs": ["r", "W2t", "b2"], "output": "y"},
        {"op": "Softmax", "inputs": ["y"], "output": "p"},
    ]
    x = rng.normal(size=(5, 4))
    ref = np.maximum(x @ inits["W1"].T + inits["b1"], 0) @ (2 * inits["W2"]) + inits["b2"]
    ref = np.exp(ref - ref.max(1, keepdims=True)); ref /= ref.sum(1, keepdims=True)

    R.test("run_graph", lambda: close(need(ns, "run_graph")(nodes, inits, {"x": x})["p"], ref))

    def fold():
        n2, i2 = need(ns, "fold_constants")(nodes, inits)
        ensure([n["op"] for n in n2] == ["Gemm", "Relu", "Gemm", "Softmax"], f"katlama sonrası {[n['op'] for n in n2]}")
        ensure("W2t" in i2 and "W2t" not in inits, "katlanan sabit yeni inits'te; orijinal sözlük değişmemeli")
        close(ns["run_graph"](n2, i2, {"x": x})["p"], ref, what="katlanmış graf aynı sonuç")
    R.test("fold_constants", fold)

    def shapes():
        sh = need(ns, "infer_shapes")(nodes, inits, {"x": ("N", 4)})
        want = {"h": ("N", 8), "r": ("N", 8), "W2s": (8, 3), "W2t": (3, 8), "y": ("N", 3), "p": ("N", 3)}
        for k, v in want.items():
            ensure(tuple(sh[k]) == v, f"{k}: {sh[k]}, beklenen {v}")
        sh2 = ns["infer_shapes"]([{"op": "Add", "inputs": ["a", "b"], "output": "c"}], {}, {"a": ("N", 1, 5), "b": (7, 1)})
        ensure(tuple(sh2["c"]) == ("N", 7, 5), f"yayınlama: {sh2['c']}")
    R.test("infer_shapes (sembolik batch, yayınlama)", shapes)
