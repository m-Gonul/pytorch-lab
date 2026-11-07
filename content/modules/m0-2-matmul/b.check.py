import random
from labcheck import close, no_imports, need


def ref(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "pandas"}))
    R.test("`matmul` tanımlı", lambda: need(ns, "matmul"))
    matmul = ns.get("matmul")
    if matmul is None:
        return
    R.test("elle örnek: [[19,22],[43,50]]", lambda: close(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]]))
    R.test("1×1", lambda: close(matmul([[3]], [[-2]]), [[-6]]))

    def rect():
        rng = random.Random(0)
        for n, k, m in [(3, 4, 2), (1, 5, 3), (4, 1, 4)]:
            A = [[rng.uniform(-2, 2) for _ in range(k)] for _ in range(n)]
            B = [[rng.uniform(-2, 2) for _ in range(m)] for _ in range(k)]
            close(matmul(A, B), ref(A, B), what=f"({n}×{k})({k}×{m})")
    R.test("dikdörtgen matrisler (3×4·4×2, 1×5·5×3, 4×1·1×4)", rect)

    def mismatch():
        try:
            matmul([[1, 2, 3]], [[1, 2], [3, 4]])
        except ValueError:
            return
        raise AssertionError("(1×3)(2×2) için ValueError bekleniyordu")
    R.test("shape uyumsuzluğunda ValueError", mismatch)

    def pure():
        A, B = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
        matmul(A, B)
        assert A == [[1, 2], [3, 4]] and B == [[5, 6], [7, 8]], "girdiler değiştirilmiş"
    R.test("girdiler değişmiyor", pure)
