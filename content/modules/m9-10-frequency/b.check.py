import time
import numpy as np
from labcheck import close as _close, no_imports, need, ensure


def close(a, b, **kw):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    _close(a.real, b.real, **kw)
    _close(a.imag, b.imag, **kw)


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "scipy", "torch"}))
    rng = np.random.default_rng(0)

    def d():
        for N in (1, 5, 8, 12):
            x = rng.normal(size=N)
            close(np.array(need(ns, "dft")(x.tolist())), np.fft.fft(x), atol=1e-9, what=f"N = {N}")
    R.test("dft = np.fft.fft", d)

    def f():
        for N in (2, 16, 64, 256, 6):
            x = rng.normal(size=N) + 1j * rng.normal(size=N)
            close(np.array(need(ns, "fft")(x.tolist())), np.fft.fft(x), atol=1e-8, what=f"N = {N}")
            close(np.array(need(ns, "idft")(np.fft.fft(x).tolist())), x, atol=1e-9, what=f"idft N = {N}")
    R.test("fft ve idft", f)

    def two():
        img = rng.normal(size=(8, 16))
        close(np.array(need(ns, "dft2")(img.tolist())), np.fft.fft2(img), atol=1e-8, what="dft2")
    R.test("dft2 = np.fft.fft2 (ayrılabilirlik)", two)

    def speed():
        x = rng.normal(size=1024).tolist()
        t0 = time.perf_counter(); ns["dft"](x[:256]); t1 = time.perf_counter(); ns["fft"](x); t2 = time.perf_counter()
        R.table("Saf Python", ["işlem", "N", "süre", "çarpma sayısı"], [["dft", 256, f"{(t1 - t0) * 1e3:.0f} ms", f"{256 ** 2:,}"], ["fft", 1024, f"{(t2 - t1) * 1e3:.0f} ms", f"≈ {512 * 10:,}"]],
                note="FFT: N/2 · log₂N kelebek çarpması. 1024 noktada DFT'nin 1/200'ü.")
    R.test("O(N²) ve O(N log N)", speed)
