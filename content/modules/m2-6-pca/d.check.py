import torch
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 4, generator=g, dtype=torch.float64) @ torch.randn(4, 4, generator=g, dtype=torch.float64)
    Xc = X - X.mean(0)
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    idx = Vh.abs().argmax(1, keepdim=True)
    Vref = Vh * Vh.gather(1, idx).sign()

    def pt():
        P, var = need(ns, "pca_torch")(X, 2)
        close(var, s[:2] ** 2 / 299, rtol=1e-8); close(P, Vref[:2], atol=1e-8)
    R.test("pca_torch = SVD (işaret kuralıyla)", pt)
    R.test("autograd kullanılıyor", lambda: ensure(uses_any(src, ["backward", "grad"]), "Rayleigh bölümünü autograd ile optimize et"))

    def gd():
        v = need(ns, "top_component_gd")(X)
        ensure(abs(float(v.norm()) - 1) < 1e-6, "birim vektör bekleniyordu")
        cos = float(v @ Vref[0])
        ensure(cos > 0.9999, f"baş bileşenle kosinüs {cos:.6f}")
        return f"cos(v, v₁) = {cos:.7f}"
    R.test("top_component_gd baş bileşene yakınsıyor", gd)
