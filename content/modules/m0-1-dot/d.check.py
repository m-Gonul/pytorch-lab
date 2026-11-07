import torch
import torch.nn.functional as F
from labcheck import close, need, uses_any, ensure


def check(ns, R):
    u = torch.tensor([1., 2., 3.], dtype=torch.float64)
    v = torch.tensor([4., -5., 6.], dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(7, generator=g, dtype=torch.float64), torch.randn(7, generator=g, dtype=torch.float64)

    def cos():
        f = need(ns, "cosine_torch")
        close(f(u, v), 12 / 1078 ** 0.5)
        close(f(a, b), F.cosine_similarity(a, b, dim=0), what="rastgele 7-boyutlu")
    R.test("cosine_torch doğru", cos)

    def grad():
        gr = need(ns, "cosine_grad")
        for x, y in ((u, v), (a, b)):
            got = gr(x, y)
            c = (x @ y) / (x.norm() * y.norm())
            want = y / (x.norm() * y.norm()) - c * x / x.norm() ** 2
            close(got, want, what="∂cos/∂u")
            ensure(abs(float(got @ x)) < 1e-9, f"gradyan u'ya dik değil: g·u = {float(got @ x):.3g}")
        return "formülle eşleşti ve gradyan u'ya dik (g·u ≈ 0)"
    R.test("cosine_grad formülle eşleşiyor, u'ya dik", grad)
    R.test("autograd kullanılıyor", lambda: ensure(uses_any(ns["__source__"], ["backward", "grad"]), "gradyanı elle yazma; .backward() kullan"))
