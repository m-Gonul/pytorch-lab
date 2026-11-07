import torch


def masked_lstm(x, lengths, W_ih, W_hh, b_ih, b_hh):
    """x: (N, T, D), lengths: (N,) → (h_T, c_T) her dizinin kendi son adımında. Yalnız zaman döngüsü."""
    N, T, _ = x.shape
    H = W_hh.shape[1]
    h = x.new_zeros(N, H); c = x.new_zeros(N, H)
    xs = x @ W_ih.T + b_ih                                        # girdi kısmı: tüm adımlar tek matmul
    for t in range(T):
        i, f, g, o = (xs[:, t] + h @ W_hh.T + b_hh).chunk(4, 1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        m = (t < lengths).to(x.dtype)[:, None]
        h = m * h_new + (1 - m) * h
        c = m * c_new + (1 - m) * c
    return h, c
