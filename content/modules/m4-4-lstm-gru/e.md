Farklı uzunlukta $N$ diziyi tek batch'te işle: `x: (N, T_max, D)` dolgulu, `lengths: (N,)`. Her adımda maske $m_t=[t<\text{len}]$ ile güncellemeyi yalnız geçerli dizilere uygula:

$$
h\leftarrow m_t\odot h_{\text{yeni}}+(1-m_t)\odot h,\qquad c\leftarrow m_t\odot c_{\text{yeni}}+(1-m_t)\odot c .
$$

- `masked_lstm(x, lengths, W_ih, W_hh, b_ih, b_hh)` → `(h_T, c_T)`: her dizinin **kendi** son geçerli adımındaki durum. Yalnız zaman döngüsü (dizi döngüsü yok). Dört kapı tek matmul.

Kontrol: her diziyi tek başına çalıştırmak ve `pack_padded_sequence` + `nn.LSTM` ile aynı sonuç; hız tablosu.
