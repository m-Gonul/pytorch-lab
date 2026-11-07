import math
import numpy as np
from labcheck import close, no_imports, need, ensure


def bf16_ref(x):
    u = np.array([x], dtype=np.float32).view(np.uint32).astype(np.uint64)
    if not np.isfinite(np.float32(x)):
        return float(np.float32(x))
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return float(u.astype(np.uint32).view(np.float32)[0])


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rf = need(ns, "round_float")

    def f16():
        rng = np.random.default_rng(0)
        xs = np.concatenate([rng.normal(size=300) * 10.0 ** rng.integers(-9, 6, 300), [65504, 65519, 65520, 7e4, 2 ** -24, 2 ** -25, 3 * 2 ** -26, 1e-8, -0.1, 1 + 2 ** -11, 1 + 3 * 2 ** -11]])
        for x in xs:
            with np.errstate(over="ignore"):
                want = float(np.float16(x))
            got = rf(float(x), 10, 5)
            ensure(got == want or (math.isinf(got) and math.isinf(want) and (got > 0) == (want > 0)), f"fp16({x!r}) = {got}, NumPy {want}")
    R.test("fp16 = NumPy float16 (taşma, subnormal, eşitlikte çift dahil)", f16)

    def b16():
        rng = np.random.default_rng(1)
        xs = np.float32(rng.normal(size=300) * 10.0 ** rng.integers(-30, 30, 300))
        for x in list(xs) + [np.float32(1 + 2 ** -8), np.float32(1 + 3 * 2 ** -8)]:
            ensure(rf(float(x), 7, 8) == bf16_ref(float(x)), f"bf16({float(x)!r}) = {rf(float(x), 7, 8)}, beklenen {bf16_ref(float(x))}")
    R.test("bf16 = bit düzeyinde yuvarlama", b16)

    def ls():
        S = need(ns, "LossScaler", "sınıf")(8.0)
        seq = []
        for inf in [False, False, False, True, False, False, False, False]:
            S.update(inf); seq.append(S.scale)
        ensure(seq == [8, 8, 16, 8, 8, 8, 16, 16], f"ölçek dizisi {seq}")
    R.test("LossScaler: büyüme ve geri çekilme", ls)
