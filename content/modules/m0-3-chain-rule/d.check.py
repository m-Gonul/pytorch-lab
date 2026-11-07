import torch
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]

    def ga():
        out = need(ns, "grads_autograd")(2.0, 3.0)
        ensure(isinstance(out, (tuple, list)) and len(out) == 2, "(df/dx, df/dy) çifti döndürmeli")
        close(out, [3, 8])
        close(need(ns, "grads_autograd")(-1.5, 0.5), [0.5, -0.5])
    R.test("grads_autograd: (3, 8) ve (−1.5, 0.5) noktası", ga)
    R.test("backward kullanılıyor", lambda: ensure(uses_any(src, ["backward", "grad"]), "türevi formülle yazma; autograd'a bırak"))

    def jt():
        J = need(ns, "jacobian_torch")(torch.tensor([2.0, 3.0]))
        ensure(torch.is_tensor(J) and tuple(J.shape) == (2, 2), "(2, 2) tensör bekleniyordu")
        close(J, [[3, 2], [1, 6]])
        close(need(ns, "jacobian_torch")(torch.tensor([-1.0, 0.5])), [[0.5, -1], [1, 1]], what="(−1, 0.5) noktası")
    R.test("jacobian_torch: [[3,2],[1,6]]", jt)
