import torch
from labcheck import close, need, uses_any, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(4, 3, generator=g, dtype=torch.float64)
    B = torch.randn(3, 5, generator=g, dtype=torch.float64)

    def mm():
        out = need(ns, "matmul_torch")(A, B)
        assert torch.is_tensor(out), "torch.Tensor döndürmeli"
        close(out, A @ B, what="(4×3)(3×5)")
    R.test("matmul_torch doğru", mm)

    def grads():
        res = need(ns, "matmul_grads")(A, B)
        assert isinstance(res, (tuple, list)) and len(res) == 2, "(dA, dB) çifti döndürmeli"
        dA, dB = res
        G = torch.ones(4, 5, dtype=torch.float64)
        close(dA, G @ B.T, what="dL/dA")
        close(dB, A.T @ G, what="dL/dB")
        return "dL/dA = 1·Bᵀ ve dL/dB = Aᵀ·1 ile eşleşti"
    R.test("gradyanlar formülle eşleşiyor", grads)

    R.test("autograd kullanılıyor (.backward / torch.autograd.grad)",
           lambda: ensure(uses_any(ns["__source__"], ["backward", "grad"]), "gradyanı elle yazma; L.backward() kullan"))
