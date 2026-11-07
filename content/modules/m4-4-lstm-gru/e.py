import torch


def masked_lstm(x, lengths, W_ih, W_hh, b_ih, b_hh):
    """x: (N, T, D), lengths: (N,) → (h_T, c_T) her dizinin kendi son adımında. Yalnız zaman döngüsü."""
    ...
