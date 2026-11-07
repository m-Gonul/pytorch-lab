import torch
from labcheck import close, need, ensure


def check(ns, R):
    A = torch.arange(12.0).reshape(3, 4)

    def lay():
        f = need(ns, "layout")
        ensure(tuple(f(A)) == ((3, 4), (4, 1), 0, True), f"A: {f(A)}")
        ensure(tuple(f(A.T)) == ((4, 3), (1, 4), 0, False), f"A.T: {f(A.T)}")
        ensure(tuple(f(A[1:, ::2])) == ((2, 2), (4, 2), 4, False), f"A[1:, ::2]: {f(A[1:, ::2])}")
        return "A[1:, ::2] → stride (4, 2), offset 4"
    R.test("layout: shape, stride, offset, contiguous", lay)

    def ss():
        f = need(ns, "same_storage")
        ensure(f(A, A.T) and f(A, A[1:, ::2]) and f(A, A.view(-1)), "view'lar aynı storage'ı paylaşmalı")
        ensure(not f(A, A.clone()) and not f(A, A.T.contiguous()), "kopyalar farklı storage")
    R.test("same_storage", ss)

    def ft():
        out = need(ns, "flatten_transposed")(A)
        close(out, A.T.contiguous().view(-1))
        ensure(out.is_contiguous(), "sonuç contiguous olmalı")
    R.test("flatten_transposed doğru", ft)

    def wt():
        x = torch.arange(20.0)
        W = need(ns, "windows_torch")(x, 4)
        ensure(tuple(W.shape) == (17, 4), f"shape {tuple(W.shape)}")
        ensure(W.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), "pencereler kopya")
        close(W, torch.stack([x[i:i + 4] for i in range(17)]))
        return f"stride = {W.stride()}"
    R.test("windows_torch: kopyasız unfold", wt)
