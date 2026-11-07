import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız zaman döngüsü", lambda: max_loops(src, 1, func="masked_lstm", why="diziler üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    lstm = nn.LSTM(3, 5, batch_first=True).double()
    P = [lstm.weight_ih_l0.detach(), lstm.weight_hh_l0.detach(), lstm.bias_ih_l0.detach(), lstm.bias_hh_l0.detach()]
    lengths = torch.tensor([7, 2, 5, 1, 7])
    x = torch.randn(5, 7, 3, generator=g, dtype=torch.float64)
    for k, L in enumerate(lengths.tolist()):
        x[k, L:] = 99.0                                     # dolgu: sızarsa sonucu bozar

    def each():
        h, c = need(ns, "masked_lstm")(x, lengths, *P)
        for k, L in enumerate(lengths.tolist()):
            _, (hk, ck) = lstm(x[k:k + 1, :L])
            close(h[k], hk[0, 0], what=f"dizi {k} h"); close(c[k], ck[0, 0], what=f"dizi {k} c")
    R.test("her dizi tek başına çalıştırılmış gibi (dolgu sızmıyor)", each)

    def packed():
        h, c = need(ns, "masked_lstm")(x, lengths, *P)
        _, (h2, c2) = lstm(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        close(h, h2[0]); close(c, c2[0])
    R.test("pack_padded_sequence + nn.LSTM ile aynı", packed)
    if "masked_lstm" not in ns:
        return
    lb = nn.LSTM(32, 64, batch_first=True)
    Pb = [p.detach() for p in (lb.weight_ih_l0, lb.weight_hh_l0, lb.bias_ih_l0, lb.bias_hh_l0)]
    Lb = torch.randint(10, 100, (64,), generator=g)
    xb = torch.randn(64, 100, 32, generator=g)
    with torch.no_grad():
        t_loop = bench(lambda: [lb(xb[k:k + 1, :int(Lb[k])]) for k in range(64)], repeat=3, number=1)
        t_vec = bench(lambda: ns["masked_lstm"](xb, Lb, *Pb), repeat=3, number=1)
    R.table("64 dizi, uzunluk 10–100", ["yöntem", "süre", "kazanç"],
            [["for dizi: nn.LSTM", fmt_time(t_loop), "1×"], ["senin masked_lstm (batch + maske)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}×"]],
            note="Maskeli batch en uzun dizinin uzunluğu kadar adım atar; kısa dizilerin dolgusu boşa hesaplanır. Uzunluğa göre gruplama (bucketing) bunu azaltır.")
