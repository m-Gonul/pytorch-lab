import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)

    def cell():
        C = need(ns, "MyLSTMCell", "sınıf")
        ref = nn.LSTMCell(4, 6).double(); my = C(4, 6).double()
        my.load_state_dict(ref.state_dict())
        x = torch.randn(3, 4, generator=g, dtype=torch.float64)
        st = (torch.randn(3, 6, generator=g, dtype=torch.float64), torch.randn(3, 6, generator=g, dtype=torch.float64))
        a, b = my(x, st), ref(x, st)
        close(a[0], b[0], what="h"); close(a[1], b[1], what="c")
        close(my(x)[0], ref(x)[0], what="state = None")
    R.test("MyLSTMCell = nn.LSTMCell", cell)

    def seq():
        C = need(ns, "MyLSTMCell", "sınıf")
        lstm = nn.LSTM(4, 6, batch_first=True).double(); my = C(4, 6).double()
        my.load_state_dict({k.replace("_l0", ""): v for k, v in lstm.state_dict().items()})
        x = torch.randn(2, 9, 4, generator=g, dtype=torch.float64)
        out, (h, c) = need(ns, "run_cell")(my, x)
        o2, (h2, c2) = lstm(x)
        close(out, o2, what="out"); close(h, h2[0], what="h_T"); close(c, c2[0], what="c_T")
        w = torch.randn_like(out)
        (out * w).sum().backward(); (o2 * w).sum().backward()
        close(my.weight_hh.grad, lstm.weight_hh_l0.grad, what="∂W_hh")
    R.test("run_cell = nn.LSTM (dizi + gradyan)", seq)
