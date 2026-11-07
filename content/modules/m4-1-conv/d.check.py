import ast
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 9, 8, generator=g, dtype=torch.float64)
    w = torch.randn(5, 3, 3, 3, generator=g, dtype=torch.float64)
    b = torch.randn(5, generator=g, dtype=torch.float64)

    def fw():
        for s, p in [(1, 0), (1, 1), (2, 1)]:
            close(need(ns, "conv2d_unfold")(x, w, b, s, p), F.conv2d(x, w, b, stride=s, padding=p), what=f"stride {s}, pad {p}")
    R.test("conv2d_unfold = F.conv2d", fw)

    def gr():
        xa, wa, ba = (t.clone().requires_grad_(True) for t in (x, w, b))
        xb, wb, bb = (t.clone().requires_grad_(True) for t in (x, w, b))
        dout = torch.randn(2, 5, 5, 4, generator=g, dtype=torch.float64)
        (need(ns, "conv2d_unfold")(xa, wa, ba, 2, 1) * dout).sum().backward()
        (F.conv2d(xb, wb, bb, stride=2, padding=1) * dout).sum().backward()
        close(xa.grad, xb.grad, what="dx"); close(wa.grad, wb.grad, what="dW"); close(ba.grad, bb.grad, what="db")
    R.test("autograd gradyanları F.conv2d ile aynı", gr)

    def ig():
        fn = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "conv2d_input_grad"), None)
        ensure(fn is not None, "`conv2d_input_grad` yok")
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        ensure(not ({"backward", "grad", "conv_transpose2d"} & names), "autograd/conv_transpose2d kullanmadan, fold ile yaz")
        for s, p in [(1, 0), (2, 1), (1, 2)]:
            xr = x.clone().requires_grad_(True)
            out = F.conv2d(xr, w, None, stride=s, padding=p)
            dout = torch.randn(out.shape, generator=g, dtype=torch.float64)
            (out * dout).sum().backward()
            close(need(ns, "conv2d_input_grad")(dout, w, x.shape, s, p), xr.grad, what=f"stride {s}, pad {p}")
        return "dx = fold(W_matᵀ · dY): col2im örtüşen pencereleri topluyor"
    R.test("conv2d_input_grad (fold) = autograd dx", ig)
