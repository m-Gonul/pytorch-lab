import ast
import torch
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]

    def noauto():
        fn = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "manual_grads"), None)
        ensure(fn is not None, "`manual_grads` yok")
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        for bad in ("backward", "autograd", "func", "requires_grad", "requires_grad_", "sum_to_size", "grad"):
            ensure(bad not in names, f"manual_grads içinde `{bad}` kullanılmamalı")
    R.test("manual_grads autograd kullanmıyor", noauto)
    g = torch.Generator().manual_seed(0)

    def ub():
        f = need(ns, "unbroadcast_torch")
        G = torch.randn(4, 2, 3, generator=g)
        close(f(G, (3,)), G.sum((0, 1))); close(f(G, (2, 1)), G.sum(0).sum(1, keepdim=True)); close(f(G, torch.Size([4, 1, 3])), G.sum(1, keepdim=True))
    R.test("unbroadcast_torch", ub)

    def mg():
        a = torch.randn(7, 5, generator=g, dtype=torch.float64); b = torch.randn(5, generator=g, dtype=torch.float64); W = torch.randn(5, 4, generator=g, dtype=torch.float64)
        da, db, dW = need(ns, "manual_grads")(a, b, W)
        A, B, Wt = (t.clone().requires_grad_(True) for t in (a, b, W))
        torch.relu((A * B + A) @ Wt).sum().backward()
        close(da, A.grad, what="da"); close(db, B.grad, what="db (unbroadcast!)"); close(dW, Wt.grad, what="dW")
        ensure(tuple(db.shape) == (5,), f"db shape {tuple(db.shape)}, (5,) bekleniyordu")
    R.test("manual_grads = autograd", mg)
