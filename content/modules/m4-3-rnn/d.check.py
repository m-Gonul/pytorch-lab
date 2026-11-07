import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)

    def eq():
        C = need(ns, "MyRNN", "sınıf")
        ref = nn.RNN(4, 6, batch_first=True).double()
        my = C(4, 6).double()
        my.load_state_dict(ref.state_dict())
        x = torch.randn(3, 7, 4, generator=g, dtype=torch.float64)
        h0 = torch.randn(1, 3, 6, generator=g, dtype=torch.float64)
        o1, h1 = my(x, h0); o2, h2 = ref(x, h0)
        close(o1, o2, what="out"); close(h1, h2, what="h_n")
        w = torch.randn_like(o1)
        (o1 * w).sum().backward(); (o2 * w).sum().backward()
        for (n, a), (_, b) in zip(my.named_parameters(), ref.named_parameters()):
            close(a.grad, b.grad, what=f"∂{n}")
        o3, _ = my(x)
        close(o3, ref(x)[0], what="h0 = None")
    R.test("MyRNN = nn.RNN (çıktı + gradyanlar)", eq)

    def van():
        torch.manual_seed(0)
        rnn = nn.RNN(8, 32, batch_first=True)
        x = torch.randn(4, 50, 8, generator=g)
        n = need(ns, "input_grad_norms")(rnn, x)
        ensure(len(n) == 50, f"50 değer bekleniyordu, {len(n)} geldi")
        ensure(n[0] < 1e-3 * n[-1], f"gradyan geçmişe doğru sönmeli: ilk {n[0]:.2e}, son {n[-1]:.2e}")
        R.table("‖∂(Σh₅₀)/∂x_t‖ — varsayılan başlatmalı nn.RNN", ["t", "norm"], [[str(t + 1), f"{n[t]:.2e}"] for t in (49, 44, 39, 29, 19, 0)])
    R.test("gradyan zamanda sönüyor", van)
